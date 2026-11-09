#pragma once

#include <aria/tensor.hpp>

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace aria {

inline TensorPtr squared_error(const TensorPtr &prediction, float target) {
  auto error = prediction + Tensor::create(Matrix(1, 1, -target));
  return error * error;
}

inline TensorPtr combined_cross_entropy(const TensorPtr &logits,
                                        std::vector<std::size_t> target_classes) {
  if (!logits || logits->data.rows() == 0 || logits->data.cols() == 0 ||
      logits->data.rows() != target_classes.size()) {
    throw std::invalid_argument(
        "Cross entropy requires nonempty scores and one target per row");
  }
  for (const auto target : target_classes) {
    if (target >= logits->data.cols()) {
      throw std::invalid_argument("Cross entropy target class is out of range");
    }
  }

  const std::size_t rows = logits->data.rows();
  const std::size_t cols = logits->data.cols();

  Matrix probabilities(rows, cols);
  float loss{};

  for (std::size_t row = 0; row < rows; ++row) {
    const auto stats = detail::softmax_row(probabilities, logits->data, row);
    const auto target = target_classes[row];
    loss += stats.log_exp_sum - (logits->data[row, target] - stats.maximum);
  }

  loss = loss / static_cast<float>(rows);

  auto out = Tensor::create(Matrix(1, 1, loss), {logits}, "cross_entropy");
  Tensor *out_raw = out.get();
  out->backward_op = [logits, target_classes = std::move(target_classes),
                      probabilities = std::move(probabilities), out_raw] {
    const std::size_t rows = probabilities.rows();
    const std::size_t cols = probabilities.cols();
    detail::require_shape(logits->data, rows, cols);
    detail::require_shape(logits->grad, rows, cols);
    detail::require_shape(out_raw->grad, 1, 1);
    const float incoming = out_raw->grad[0, 0];
    const float batch_rows = static_cast<float>(rows);
    for (std::size_t row = 0; row < rows; ++row) {
      for (std::size_t col = 0; col < cols; ++col) {
        float derivative = probabilities[row, col];
        if (col == target_classes[row]) {
          derivative -= 1.0f;
        }
        logits->grad[row, col] += derivative * incoming / batch_rows;
      }
    }
  };
  return out;
}

} // namespace aria
