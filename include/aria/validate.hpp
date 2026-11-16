#pragma once

#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>

namespace aria::validate {

// these helpers depend only on dimensions or the rows()/cols() interface
struct MatrixShape {
  std::size_t rows;
  std::size_t cols;
};

inline void require_equal(std::size_t actual, std::size_t expected,
                          const char *message) {
  if (actual != expected) {
    throw std::invalid_argument(message);
  }
}

template <typename MatrixLike>
inline void require_shape(
    const MatrixLike &value, std::size_t rows, std::size_t cols,
    const char *message = "Matrix shape must match the required dimensions") {
  if (value.rows() != rows || value.cols() != cols) {
    throw std::invalid_argument(message);
  }
}

template <typename MatrixLike>
inline void require_square(const MatrixLike &value, const char *message) {
  require_equal(value.rows(), value.cols(), message);
}

inline void require_positive_dimensions(std::size_t rows, std::size_t cols,
                                        const char *message) {
  if (rows == 0 || cols == 0) {
    throw std::invalid_argument(message);
  }
}

template <typename MatrixLike>
inline void require_nonempty(const MatrixLike &value, const char *message) {
  require_positive_dimensions(value.rows(), value.cols(), message);
}

template <typename Pointer>
inline void require_nonnull(const Pointer &value, const char *message) {
  if (!value) {
    throw std::invalid_argument(message);
  }
}

inline void require_positive_finite(float value, const char *message) {
  if (!std::isfinite(value) || value <= 0.0f) {
    throw std::invalid_argument(message);
  }
}

inline void require_index(std::size_t index, std::size_t bound,
                          const char *message) {
  if (index >= bound) {
    throw std::invalid_argument(message);
  }
}

inline void require_indices(std::span<const std::size_t> indices,
                            std::size_t bound, const char *message) {
  for (const auto index : indices) {
    require_index(index, bound, message);
  }
}

inline void require_bounds(std::size_t row, std::size_t col,
                           std::size_t rows, std::size_t cols) {
  if (row >= rows || col >= cols) {
    throw std::out_of_range("Matrix row or column out of range");
  }
}

inline std::size_t element_count(std::size_t rows, std::size_t cols) {
  if (cols != 0 && rows > std::numeric_limits<std::size_t>::max() / cols) {
    throw std::length_error("Matrix dimensions overflow the element count");
  }
  return rows * cols;
}

template <typename MatrixLike>
inline void require_matmul(const MatrixLike &a, const MatrixLike &b) {
  require_equal(a.cols(), b.rows(), "Matrix A cols must equal Matrix B rows");
}

template <typename MatrixLike>
inline void require_separate(const MatrixLike &out, const MatrixLike &in,
                             const char *message) {
  if (&out == &in) {
    throw std::invalid_argument(message);
  }
}

template <typename MatrixLike>
inline MatrixShape broadcast_shape(const MatrixLike &a, const MatrixLike &b) {
  const bool rows_ok = a.rows() == b.rows() || a.rows() == 1 || b.rows() == 1;
  const bool cols_ok = a.cols() == b.cols() || a.cols() == 1 || b.cols() == 1;
  if (!rows_ok || !cols_ok) {
    throw std::invalid_argument("Incompatible matrix shapes for broadcasting");
  }
  return {a.rows() == 1 ? b.rows() : a.rows(),
          a.cols() == 1 ? b.cols() : a.cols()};
}

inline std::size_t add_columns(std::size_t left, std::size_t right) {
  if (right > std::numeric_limits<std::size_t>::max() - left) {
    throw std::length_error("Concatenated column count overflows");
  }
  return left + right;
}

template <typename MatrixLike>
inline MatrixShape concat_columns_shape(std::span<MatrixLike *const> inputs) {
  require_positive_dimensions(1, inputs.size(),
                               "Column concatenation requires at least one input");
  require_nonnull(inputs.front(), "Column concatenation inputs must not be null");
  const std::size_t rows = inputs.front()->rows();
  std::size_t cols = 0;
  for (const auto *input : inputs) {
    require_nonnull(input, "Column concatenation inputs must not be null");
    require_equal(input->rows(), rows,
                   "Column concatenation inputs must have equal row counts");
    cols = add_columns(cols, input->cols());
  }
  return {rows, cols};
}

} // namespace aria::validate

