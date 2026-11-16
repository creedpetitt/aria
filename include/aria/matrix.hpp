#pragma once

#include <aria/validate.hpp>

#ifdef ARIA_USE_MKL
#include <aria/backend.hpp>
#endif

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <ostream>
#include <span>
#include <utility>
#include <vector>

namespace aria {

class Matrix {
public:
  Matrix() = default;

  Matrix(std::size_t rows, std::size_t cols, float data = 0.0f)
      : rows_(rows), cols_(cols), data_(validate::element_count(rows, cols), data) {}

  static Matrix from_buffer(std::size_t rows, std::size_t cols,
                            std::vector<float> values) {
    validate::require_equal(values.size(), validate::element_count(rows, cols),
                            "Matrix buffer size must match its dimensions");
    Matrix result;
    result.rows_ = rows;
    result.cols_ = cols;
    result.data_ = std::move(values);
    return result;
  }

  Matrix(std::initializer_list<std::initializer_list<float>> rows)
      : rows_(rows.size()), cols_(rows.size() == 0 ? 0 : rows.begin()->size()) {
    for (const auto &row : rows) {
      validate::require_equal(row.size(), cols_, "Matrix rows must have equal lengths");
    }

    data_.reserve(validate::element_count(rows_, cols_));
    for (const auto &row : rows) {
      data_.insert(data_.end(), row.begin(), row.end());
    }
  }

  ~Matrix() = default;
  Matrix(const Matrix &) = default;

  Matrix &operator=(const Matrix &other) {
    if (this != &other) {
      // copy first so a failed allocation leaves this matrix unchanged
      Matrix copy(other);
      swap(copy);
    }
    return *this;
  }

  Matrix(Matrix &&other) noexcept
      : rows_(std::exchange(other.rows_, 0)),
        cols_(std::exchange(other.cols_, 0)), 
        data_(std::move(other.data_)) {
    other.data_.clear();
  }

  Matrix &operator=(Matrix &&other) noexcept {
    if (this != &other) {
      data_ = std::move(other.data_);
      rows_ = std::exchange(other.rows_, 0);
      cols_ = std::exchange(other.cols_, 0);
      other.data_.clear();
    }
    return *this;
  }

  void swap(Matrix &other) noexcept {
    std::swap(rows_, other.rows_);
    std::swap(cols_, other.cols_);
    data_.swap(other.data_);
  }

  float &operator[](std::size_t row, std::size_t col) {
    validate::require_bounds(row, col, rows_, cols_);
    return data_[row * cols_ + col];
  }

  const float &operator[](std::size_t row, std::size_t col) const {
    validate::require_bounds(row, col, rows_, cols_);
    return data_[row * cols_ + col];
  }

  void fill(float value) { std::fill(data_.begin(), data_.end(), value); }

  std::size_t rows() const noexcept { return rows_; }
  std::size_t cols() const noexcept { return cols_; }

  float *data() noexcept { return data_.data(); }
  const float *data() const noexcept { return data_.data(); }

  friend std::ostream &operator<<(std::ostream &out, const Matrix &value) {
    if (value.rows_ == 0) {
      return out << "[]";
    }
    for (std::size_t row = 0; row < value.rows_; ++row) {
      if (row != 0) {
        out << '\n';
      }
      out << '[';
      for (std::size_t col = 0; col < value.cols_; ++col) {
        if (col != 0) {
          out << ' ';
        }
        out << value.data_[row * value.cols_ + col];
      }
      out << ']';
    }
    return out;
  }

private:
  std::size_t rows_{};
  std::size_t cols_{};
  std::vector<float> data_;
};

namespace detail {

template <typename Operation>
inline void broadcast_elementwise(Matrix &out, const Matrix &a, const Matrix &b,
                                  Operation operation) {
  const auto shape = validate::broadcast_shape(a, b);
  validate::require_shape(out, shape.rows, shape.cols);
  if (shape.rows == 0 || shape.cols == 0) {
    return;
  }
  float *const out_data = out.data();
  const float *const a_data = a.data();
  const float *const b_data = b.data();

  if (a.rows() == b.rows() && a.cols() == b.cols()) {
    const std::size_t count = shape.rows * shape.cols;
    for (std::size_t i = 0; i < count; ++i) {
      out_data[i] = operation(a_data[i], b_data[i]);
    }
    return;
  }

  const std::size_t a_row_stride = a.rows() == 1 ? 0 : a.cols();
  const std::size_t b_row_stride = b.rows() == 1 ? 0 : b.cols();
  const std::size_t a_col_stride = a.cols() == 1 ? 0 : 1;
  const std::size_t b_col_stride = b.cols() == 1 ? 0 : 1;
  for (std::size_t row = 0; row < shape.rows; ++row) {
    const std::size_t out_offset = row * shape.cols;
    const std::size_t a_offset = row * a_row_stride;
    const std::size_t b_offset = row * b_row_stride;
    for (std::size_t col = 0; col < shape.cols; ++col) {
      const float a_value = a_data[a_offset + col * a_col_stride];
      const float b_value = b_data[b_offset + col * b_col_stride];
      out_data[out_offset + col] = operation(a_value, b_value);
    }
  }
}

struct SoftmaxRowStats {
  float maximum;
  float log_exp_sum;
};

inline SoftmaxRowStats softmax_row(Matrix &out, const Matrix &in,
                                  std::size_t row) {
  const std::size_t cols = in.cols();
  const float *const in_row = in.data() + row * cols;
  float *const out_row = out.data() + row * cols;
  float maximum = in_row[0];
  for (std::size_t col = 1; col < cols; ++col) {
    maximum = std::max(maximum, in_row[col]);
  }

  float exp_sum = 0.0f;
  for (std::size_t col = 0; col < cols; ++col) {
    const float value = std::exp(in_row[col] - maximum);
    out_row[col] = value;
    exp_sum += value;
  }

  for (std::size_t col = 0; col < cols; ++col) {
    out_row[col] /= exp_sum;
  }

  return {maximum, std::log(exp_sum)};
}

} // namespace detail

inline void softmax(Matrix &out, const Matrix &in) {
  const std::size_t rows = in.rows();
  const std::size_t cols = in.cols();
  validate::require_shape(out, rows, cols);
  if (cols == 0) {
    return;
  }
  for (std::size_t row = 0; row < rows; ++row) {
    detail::softmax_row(out, in, row);
  }
}

inline Matrix softmax(const Matrix &in) {
  Matrix out(in.rows(), in.cols());
  softmax(out, in);
  return out;
}

inline void add(Matrix &out, const Matrix &a, const Matrix &b) {
  detail::broadcast_elementwise(out, a, b,
                                [](float x, float y) { return x + y; });
}

inline void subtract(Matrix &out, const Matrix &a, const Matrix &b) {
  detail::broadcast_elementwise(out, a, b,
                                [](float x, float y) { return x - y; });
}

inline void multiply(Matrix &out, const Matrix &a, const Matrix &b) {
  validate::require_shape(b, a.rows(), a.cols());
  validate::require_shape(out, a.rows(), a.cols());
  float *const out_data = out.data();
  const float *const a_data = a.data();
  const float *const b_data = b.data();
  const std::size_t count = a.rows() * a.cols();
  for (std::size_t i = 0; i < count; ++i) {
    out_data[i] = a_data[i] * b_data[i];
  }
}

inline void scale(Matrix &out, const Matrix &a, float scalar) {
  validate::require_shape(out, a.rows(), a.cols());
  float *const out_data = out.data();
  const float *const a_data = a.data();
  const std::size_t count = a.rows() * a.cols();
  for (std::size_t i = 0; i < count; ++i) {
    out_data[i] = a_data[i] * scalar;
  }
}

inline void causal_mask(Matrix &out, const Matrix &in) {
  validate::require_square(in, "Causal mask requires a square score matrix");
  validate::require_shape(out, in.rows(), in.cols(),
                          "Causal mask output must match the input shape");

  const float blocked = -(std::numeric_limits<float>::infinity());

  for (std::size_t row = 0; row < in.rows(); ++row) {
    for (std::size_t col = 0; col < in.cols(); ++col) {
      out[row, col] = col <= row ? in[row, col] : blocked;
    }
  }
}

inline Matrix operator*(const Matrix &a, const Matrix &b) {
  validate::require_shape(b, a.rows(), a.cols());
  Matrix out(a.rows(), a.cols());
  multiply(out, a, b);
  return out;
}

inline Matrix operator*(const Matrix &a, float scalar) {
  Matrix out(a.rows(), a.cols());
  scale(out, a, scalar);
  return out;
}

inline Matrix operator+(const Matrix &a, const Matrix &b) {
  const auto shape = validate::broadcast_shape(a, b);
  Matrix out(shape.rows, shape.cols);
  add(out, a, b);
  return out;
}

inline Matrix operator-(const Matrix &a, const Matrix &b) {
  const auto shape = validate::broadcast_shape(a, b);
  Matrix out(shape.rows, shape.cols);
  subtract(out, a, b);
  return out;
}

inline Matrix &operator+=(Matrix &a, const Matrix &b) {
  add(a, a, b);
  return a;
}

inline Matrix &operator-=(Matrix &a, const Matrix &b) {
  subtract(a, a, b);
  return a;
}

inline Matrix &operator*=(Matrix &a, const Matrix &b) {
  multiply(a, a, b);
  return a;
}

inline Matrix &operator*=(Matrix &a, float scalar) {
  scale(a, a, scalar);
  return a;
}

inline void matmul(Matrix &out, const Matrix &a, const Matrix &b) {
#ifdef ARIA_USE_MKL
  backend::gemm(out, a, b);
#else
  validate::require_matmul(a, b);
  const std::size_t rows = a.rows();
  const std::size_t inner = a.cols();
  const std::size_t cols = b.cols();
  validate::require_shape(out, rows, cols);
  validate::require_separate(out, a, "Matmul output must be separate from its inputs");
  validate::require_separate(out, b, "Matmul output must be separate from its inputs");
  out.fill(0.0f);
  if (rows == 0 || cols == 0 || inner == 0) {
    return;
  }

  float *const out_data = out.data();
  const float *const a_data = a.data();
  const float *const b_data = b.data();
  // reuse sections of B across output rows
  constexpr std::size_t block = 64;
#ifdef ARIA_USE_OPENMP
#pragma omp parallel for schedule(static) if(rows > block && inner >= block && cols >= block)
#endif
  for (std::size_t ii = 0; ii < rows; ii += block) {
    for (std::size_t kk = 0; kk < inner; kk += block) {
      for (std::size_t jj = 0; jj < cols; jj += block) {
        const std::size_t i_end = std::min(ii + block, rows);
        const std::size_t k_end = std::min(kk + block, inner);
        const std::size_t j_end = std::min(jj + block, cols);
        for (std::size_t i = ii; i < i_end; ++i) {
          for (std::size_t k = kk; k < k_end; ++k) {
            const float a_val = a_data[i * inner + k];
            for (std::size_t j = jj; j < j_end; ++j) {
              out_data[i * cols + j] += a_val * b_data[k * cols + j];
            }
          }
        }
      }
    }
  }
#endif
}

inline Matrix matmul(const Matrix &a, const Matrix &b) {
  validate::require_matmul(a, b);
  Matrix out(a.rows(), b.cols());
  matmul(out, a, b);
  return out;
}

inline void transpose(Matrix &out, const Matrix &in) {
  const std::size_t rows = in.rows();
  const std::size_t cols = in.cols();
  validate::require_shape(out, cols, rows);
  validate::require_separate(out, in, "Transpose output must be separate from its input");
  if (rows == 0 || cols == 0) {
    return;
  }
  float *const out_data = out.data();
  const float *const in_data = in.data();
  for (std::size_t col = 0; col < cols; ++col) {
    for (std::size_t row = 0; row < rows; ++row) {
      out_data[col * rows + row] = in_data[row * cols + col];
    }
  }
}

inline Matrix transpose(const Matrix &in) {
  Matrix out(in.cols(), in.rows());
  transpose(out, in);
  return out;
}

inline void concat_columns(Matrix &out, std::span<const Matrix *const> inputs) {
  const auto shape = validate::concat_columns_shape(inputs);
  validate::require_shape(out, shape.rows, shape.cols);
  for (const auto *input : inputs) {
    validate::require_separate(out, *input,
                               "Column concatenation output must be separate from its inputs");
  }
  if (shape.rows == 0 || shape.cols == 0) {
    return;
  }

  float *const destination = out.data();
  std::size_t offset = 0;
  for (const auto *input : inputs) {
    const std::size_t width = input->cols();
    if (width != 0) {
      const float *const source = input->data();
      for (std::size_t row = 0; row < shape.rows; ++row) {
        std::copy_n(source + row * width, width,
                    destination + row * shape.cols + offset);
      }
    }
    offset += width;
  }
}

inline Matrix concat_columns(std::span<const Matrix *const> inputs) {
  const auto shape = validate::concat_columns_shape(inputs);
  Matrix out(shape.rows, shape.cols);
  concat_columns(out, inputs);
  return out;
}

inline Matrix causal_mask(const Matrix &in) {
  validate::require_square(in, "Causal mask requires a square score matrix");

  Matrix out(in.rows(), in.cols());
  causal_mask(out, in);
  return out;
}

} // namespace aria
