#pragma once

#include <algorithm>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace htm {

class matrix {
public:
  matrix() = default;

  matrix(std::size_t rows, std::size_t cols, float data = 0.0f)
      : rows_(rows), cols_(cols), data_(element_count(rows, cols), data) {}

  matrix(std::initializer_list<std::initializer_list<float>> rows)
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

  ~matrix() = default;

  // Copy Ctor and Assignment
  matrix(const matrix &) = default;

  matrix &operator=(const matrix &other) {
    if (this != &other) {
      // Copy first so a failed allocation leaves this matrix unchanged.
      matrix copy(other);
      swap(copy);
    }
    return *this;
  }

  // Move Ctor and Assignment
  matrix(matrix &&other) noexcept
      : rows_(std::exchange(other.rows_, 0)),
        cols_(std::exchange(other.cols_, 0)), data_(std::move(other.data_)) {
    other.data_.clear();
  }

  matrix &operator=(matrix &&other) noexcept {
    if (this != &other) {
      data_ = std::move(other.data_);
      rows_ = std::exchange(other.rows_, 0);
      cols_ = std::exchange(other.cols_, 0);
      other.data_.clear();
    }
    return *this;
  }

  void swap(matrix &other) noexcept {
    std::swap(rows_, other.rows_);
    std::swap(cols_, other.cols_);
    data_.swap(other.data_);
  }

  float &access_elem(std::size_t row, std::size_t col) {
    if (row >= rows_ || col >= cols_) {
      throw std::out_of_range("Matrix row or column out of range");
    }
    return data_.at(row * cols_ + col);
  };

  const float &access_elem(std::size_t row, std::size_t col) const {
    if (row >= rows_ || col >= cols_) {
      throw std::out_of_range("Matrix row or column out of range");
    }
    return data_.at(row * cols_ + col);
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

  friend std::ostream &operator<<(std::ostream &out, const matrix &value) {
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

  void add(matrix &c, const matrix &a, const matrix &b) {
    if (a.rows_ != b.rows_ || a.cols_ != b.cols_ || c.rows_ != a.rows_ ||
        c.cols_ != a.cols_) {
      throw std::invalid_argument(
          "Matrices must be same size to perform addition");
    }

    for (std::size_t i{}; i < a.data_.size(); ++i) {
      c.data_[i] = a.data_[i] + b.data_[i];
    }
  };

  void elem_mul(matrix &c, const matrix &a, const matrix &b) {
    if (a.rows_ != b.rows_ || a.cols_ != b.cols_ || c.rows_ != a.rows_ ||
        c.cols_ != a.cols_) {
      throw std::invalid_argument(
          "Matrices must be same size to perform element multiplication");
    }

    for (std::size_t i{}; i < a.data_.size(); ++i) {
      c.data_[i] = a.data_[i] * b.data_[i];
    }
  }

  void matmul(matrix &c, const matrix &a, const matrix &b) {
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

  void scalar_mul(matrix &out, const matrix &in, float scalar) {
    if (out.rows() != in.rows() || out.cols() != in.cols()) {
      throw std::invalid_argument("Matrices need to be the same size");
    }

    for (std::size_t i{}; i < in.rows() * in.cols(); ++i) {
      out.data_[i] = in.data_[i] * scalar;
    }
  }

  void transpose(matrix &out, const matrix &in) {
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
        out.access_elem(i, j) = in.access_elem(j, i);
      }
    }
  }

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

inline matrix operator+(const matrix &a, const matrix &b) {
  matrix c(a.rows(), a.cols());
  c.add(c, a, b);
  return c;
}

inline matrix operator*(const matrix &a, const matrix &b) {
  matrix c(a.rows(), a.cols());
  c.elem_mul(c, a, b);
  return c;
}
inline matrix operator*(const matrix &a, const float scalar) {
  matrix c(a.rows(), a.cols());
  c.scalar_mul(c, a, scalar);
  return c;
}

} // namespace htm
