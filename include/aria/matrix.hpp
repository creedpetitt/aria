#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace aria {

class Matrix {
public:
  Matrix() = default;

  Matrix(std::size_t rows, std::size_t cols, float data = 0.0f)
      : rows_(rows), cols_(cols), data_(element_count(rows, cols), data) {}

  static Matrix from_buffer(std::size_t rows, std::size_t cols,
                            std::vector<float> values) {
    if (values.size() != element_count(rows, cols)) {
      throw std::invalid_argument(
          "Matrix buffer size must match its dimensions");
    }
    Matrix result;
    result.rows_ = rows;
    result.cols_ = cols;
    result.data_ = std::move(values);
    return result;
  }

  Matrix(std::initializer_list<std::initializer_list<float>> rows)
      : rows_(rows.size()), cols_(rows.size() == 0 ? 0 : rows.begin()->size()) {
    for (const auto &row : rows) {
      if (row.size() != cols_) {
        throw std::invalid_argument("Matrix rows must have equal lengths");
      }
    }

    data_.reserve(element_count(rows_, cols_));
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
        cols_(std::exchange(other.cols_, 0)), data_(std::move(other.data_)) {
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

  float &access_elem(std::size_t row, std::size_t col) {
    if (row >= rows_ || col >= cols_) {
      throw std::out_of_range("Matrix row or column out of range");
    }
    return data_[row * cols_ + col];
  };

  const float &access_elem(std::size_t row, std::size_t col) const {
    if (row >= rows_ || col >= cols_) {
      throw std::out_of_range("Matrix row or column out of range");
    }
    return data_.at(row * cols_ + col);
  }

  float &operator[](std::size_t row, std::size_t col) {
    return access_elem(row, col);
  }
  const float &operator[](std::size_t row, std::size_t col) const {
    return access_elem(row, col);
  }

  bool is_zero() const {
    for (float value : data_) {
      if (value != 0.0f) {
        return false;
      }
    }
    return true;
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
  static std::size_t element_count(std::size_t rows, std::size_t cols) {
    if (cols != 0 && rows > std::numeric_limits<std::size_t>::max() / cols) {
      throw std::length_error("Matrix dimensions overflow the element count");
    }
    return rows * cols;
  }

  std::size_t rows_{};
  std::size_t cols_{};
  std::vector<float> data_;
};

namespace detail {

struct SoftmaxRowStats {
  float maximum;
  float log_exp_sum;
};

inline SoftmaxRowStats softmax_row(Matrix &out, const Matrix &in,
                                   std::size_t row) {
  float maximum = in.access_elem(row, 0);
  for (std::size_t col = 1; col < in.cols(); ++col) {
    maximum = std::max(maximum, in.access_elem(row, col));
  }

  float exp_sum = 0.0f;
  for (std::size_t col = 0; col < in.cols(); ++col) {
    float value = std::exp(in.access_elem(row, col) - maximum);
    out.access_elem(row, col) = value;
    exp_sum += value;
  }

  for (std::size_t col = 0; col < in.cols(); ++col) {
    out.access_elem(row, col) /= exp_sum;
  }

  return {maximum, std::log(exp_sum)};
}

} // namespace detail

inline Matrix softmax(const Matrix &in) {
  Matrix out(in.rows(), in.cols());
  if (in.cols() == 0) {
    return out;
  }
  for (std::size_t row = 0; row < in.rows(); ++row) {
    detail::softmax_row(out, in, row);
  }
  return out;
}

inline Matrix operator*(const Matrix &a, const Matrix &b) {
  if (a.rows() != b.rows() || a.cols() != b.cols()) {
    throw std::invalid_argument(
        "Matrices must be same size to perform element multiplication");
  }
  Matrix out(a.rows(), a.cols());
  for (std::size_t i = 0; i < a.rows() * a.cols(); ++i) {
    out.data()[i] = a.data()[i] * b.data()[i];
  }
  return out;
}

inline Matrix operator*(const Matrix &a, const float scalar) {
  Matrix out(a.rows(), a.cols());
  for (std::size_t i = 0; i < a.rows() * a.cols(); ++i) {
    out.data()[i] = a.data()[i] * scalar;
  }
  return out;
}

inline Matrix operator+(const Matrix &a, const Matrix &b) {
  bool rows_ok = a.rows() == b.rows() || a.rows() == 1 || b.rows() == 1;
  bool cols_ok = a.cols() == b.cols() || a.cols() == 1 || b.cols() == 1;
  if (!rows_ok || !cols_ok) {
    throw std::invalid_argument("Incompatable shapes");
  }

  auto rows = a.rows();
  auto cols = a.cols();

  if (rows == 1) {
    rows = b.rows();
  }
  if (cols == 1) {
    cols = b.cols();
  }

  Matrix out(rows, cols);
  for (std::size_t row{}; row < rows; ++row) {
    for (std::size_t col{}; col < cols; ++col) {
      auto a_row = row;
      auto a_col = col;
      auto b_row = row;
      auto b_col = col;

      if (a.rows() == 1)
        a_row = 0;
      if (a.cols() == 1)
        a_col = 0;

      if (b.rows() == 1)
        b_row = 0;
      if (b.cols() == 1)
        b_col = 0;

      out.data()[row * cols + col] = a.data()[a_row * a.cols() + a_col] +
                                     b.data()[b_row * b.cols() + b_col];
    }
  }
  return out;
}

inline Matrix operator-(const Matrix &a, const Matrix &b) {
  return a + (b * -1.0f);
}

inline Matrix matmul(const Matrix &a, const Matrix &b) {
  if (a.cols() != b.rows()) {
    throw std::invalid_argument("Matrix A cols must equal Matrix B rows");
  }
  Matrix out(a.rows(), b.cols());
  for (std::size_t i = 0; i < out.rows(); ++i) {
    for (std::size_t k = 0; k < a.cols(); ++k) {
      const float a_val = a.data()[i * a.cols() + k];
      for (std::size_t j = 0; j < out.cols(); ++j) {
        out.data()[i * out.cols() + j] += a_val * b.data()[k * b.cols() + j];
      }
    }
  }
  return out;
}

inline Matrix transpose(const Matrix &in) {
  Matrix out(in.cols(), in.rows());
  for (std::size_t col = 0; col < in.cols(); ++col) {
    for (std::size_t row = 0; row < in.rows(); ++row) {
      out.data()[col * out.cols() + row] = in.data()[row * in.cols() + col];
    }
  }
  return out;
}

} // namespace aria
