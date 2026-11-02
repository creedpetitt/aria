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

  // Copy Ctor and Assignment
  Matrix(const Matrix &) = default;

  Matrix &operator=(const Matrix &other) {
    if (this != &other) {
      // Copy first so a failed allocation leaves this matrix unchanged.
      Matrix copy(other);
      swap(copy);
    }
    return *this;
  }

  // Move Ctor and Assignment
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

  void add(Matrix &c, const Matrix &a, const Matrix &b) {
    if (a.rows_ != b.rows_ || a.cols_ != b.cols_ || c.rows_ != a.rows_ ||
        c.cols_ != a.cols_) {
      throw std::invalid_argument(
          "Matrices must be same size to perform addition");
    }

    for (std::size_t i{}; i < a.data_.size(); ++i) {
      c.data_[i] = a.data_[i] + b.data_[i];
    }
  };

  void elem_mul(Matrix &c, const Matrix &a, const Matrix &b) {
    if (a.rows_ != b.rows_ || a.cols_ != b.cols_ || c.rows_ != a.rows_ ||
        c.cols_ != a.cols_) {
      throw std::invalid_argument(
          "Matrices must be same size to perform element multiplication");
    }

    for (std::size_t i{}; i < a.data_.size(); ++i) {
      c.data_[i] = a.data_[i] * b.data_[i];
    }
  }

  void matmul(Matrix &c, const Matrix &a, const Matrix &b) {
    if (a.cols_ != b.rows_) {
      throw std::invalid_argument("Matrix A cols must equal Matrix B rows");
    }
    if (c.rows_ != a.rows_ || c.cols_ != b.cols_) {
      throw std::invalid_argument(
          "Matmul output must have A rows and B columns");
    }
    if (&c == &a || &c == &b) {
      throw std::invalid_argument("Matmul requires a separate output matrix");
    }

    std::fill(c.data_.begin(), c.data_.end(), 0.0f);

    for (std::size_t i{}; i < c.rows_; ++i) {
      for (std::size_t k{}; k < a.cols_; ++k) {
        float a_val = a.data_[i * a.cols_ + k];
        for (std::size_t j{}; j < c.cols_; ++j) {
          c.data_[i * c.cols_ + j] += a_val * b.data_[k * b.cols_ + j];
        }
      }
    }
  }

  void scalar_mul(Matrix &out, const Matrix &in, float scalar) {
    if (out.rows() != in.rows() || out.cols() != in.cols()) {
      throw std::invalid_argument("Matrices need to be the same size");
    }

    for (std::size_t i{}; i < in.rows() * in.cols(); ++i) {
      out.data_[i] = in.data_[i] * scalar;
    }
  }

  void transpose(Matrix &out, const Matrix &in) {
    if (out.rows_ != in.cols_ || out.cols_ != in.rows_) {
      throw std::invalid_argument(
          "Transpose output must have reversed input dimensions");
    }
    if (&out == &in) {
      throw std::invalid_argument(
          "Transpose requires separate input and output matrices");
    }

    for (std::size_t i{}; i < in.cols_; ++i) {
      for (std::size_t j{}; j < in.rows_; ++j) {
        out.data_[i * out.cols_ + j] = in.data_[j * in.cols_ + i];
      }
    }
  }

  void softmax(Matrix &out, const Matrix &in);

  bool same_size(std::size_t a, std::size_t b) { return a == b; }

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

// Callers validate matching shapes and a nonempty row before calling.
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

inline void Matrix::softmax(Matrix &out, const Matrix &in) {
  if (out.rows() != in.rows() || out.cols() != in.cols()) {
    throw std::invalid_argument("Softmax input and output shapes must match");
  }
  if (in.cols() == 0) {
    return;
  }
  for (std::size_t row = 0; row < in.rows(); ++row) {
    detail::softmax_row(out, in, row);
  }
}

inline Matrix operator+(const Matrix &a, const Matrix &b) {
  Matrix c(a.rows(), a.cols());
  c.add(c, a, b);
  return c;
}

inline void matmul(Matrix &out, const Matrix &a, const Matrix &b) {
  out.matmul(out, a, b);
}

inline Matrix matmul(const Matrix &a, const Matrix &b) {
  Matrix out(a.rows(), b.cols());
  matmul(out, a, b);
  return out;
}

inline Matrix operator*(const Matrix &a, const Matrix &b) {
  Matrix c(a.rows(), a.cols());
  c.elem_mul(c, a, b);
  return c;
}

inline Matrix operator*(const Matrix &a, const float scalar) {
  Matrix c(a.rows(), a.cols());
  c.scalar_mul(c, a, scalar);
  return c;
}

inline void transpose(Matrix &out, const Matrix &in) { out.transpose(out, in); }

inline Matrix transpose(const Matrix &in) {
  Matrix out(in.cols(), in.rows());
  transpose(out, in);
  return out;
}

} // namespace aria
