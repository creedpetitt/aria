#pragma once

#include <aria/matrix.hpp>

#include <cstddef>
#include <stdexcept>

namespace aria {
  enum class AutogradOp {
    Matmul,
    Add,
    Multiply,
    Scale,
    Transpose,
    Softmax,
    ReLU,
    MSE,
    CrossEntropy, 
    None
  };
}

namespace aria::autograd {

inline void matmul_backward(Matrix &a_grad, Matrix &b_grad, const Matrix &a,
                            const Matrix &b, const Matrix &incoming) {
  a_grad += matmul(incoming, transpose(b));
  b_grad += matmul(transpose(a), incoming);
}

inline void add_backward(Matrix &a_grad, Matrix &b_grad, const Matrix &a,
                         const Matrix &b, const Matrix &incoming) {
  const auto shape = detail::broadcast_shape(a, b);
  detail::require_shape(incoming, shape.rows, shape.cols);
  detail::require_shape(a_grad, a.rows(), a.cols());
  detail::require_shape(b_grad, b.rows(), b.cols());
  if (shape.rows == 0 || shape.cols == 0) {
    return;
  }

  // sum broadcast contributions into each inputs original row and column
  for (std::size_t row = 0; row < shape.rows; ++row) {
    const std::size_t a_row = a.rows() == 1 ? 0 : row;
    const std::size_t b_row = b.rows() == 1 ? 0 : row;
    for (std::size_t col = 0; col < shape.cols; ++col) {
      const std::size_t a_col = a.cols() == 1 ? 0 : col;
      const std::size_t b_col = b.cols() == 1 ? 0 : col;
      const float value = incoming[row, col];
      a_grad[a_row, a_col] += value;
      b_grad[b_row, b_col] += value;
    }
  }
}

inline void multiply_backward(Matrix &a_grad, Matrix &b_grad, const Matrix &a,
                              const Matrix &b, const Matrix &incoming) {
  a_grad += b * incoming;
  b_grad += a * incoming;
}

inline void scale_backward(Matrix &input_grad, const Matrix &incoming,
                           float scalar) {
  input_grad += incoming * scalar;
}

inline void transpose_backward(Matrix &input_grad, const Matrix &incoming) {
  input_grad += transpose(incoming);
}

inline void softmax_backward(Matrix &input_grad, const Matrix &probabilities,
                             const Matrix &incoming) {
  const std::size_t rows = probabilities.rows();
  const std::size_t cols = probabilities.cols();
  if (input_grad.rows() != rows || input_grad.cols() != cols ||
      incoming.rows() != rows || incoming.cols() != cols) {
    throw std::invalid_argument(
        "Softmax gradients must match the probability dimensions");
  }

  for (std::size_t row = 0; row < rows; ++row) {
    float dot = 0.0f;
    for (std::size_t col = 0; col < cols; ++col) {
      dot += incoming[row, col] * probabilities[row, col];
    }

    // apply the softmax jacobian without constructing a jacobian matrix
    for (std::size_t col = 0; col < cols; ++col) {
      input_grad[row, col] +=
          probabilities[row, col] * (incoming[row, col] - dot);
    }
  }
}

inline void relu_backward(Matrix &input_grad, const Matrix &input,
                          const Matrix &incoming) {
  const std::size_t rows = input.rows();
  const std::size_t cols = input.cols();
  detail::require_shape(input_grad, rows, cols);
  detail::require_shape(incoming, rows, cols);
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t col = 0; col < cols; ++col) {
      if (input[row, col] > 0.0f) {
        input_grad[row, col] += incoming[row, col];
      }
    }
  }
}

} // namespace aria::autograd
