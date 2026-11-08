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

  const size_t rows = logits->data.rows();
  const size_t cols = logits->data.cols();

  Matrix probabilities(rows, cols);
  float loss{};

  for (size_t row{}; row < rows; row++) {
    const auto stats = detail::softmax_row(probabilities, logits->data, row);
    const auto target = target_classes[row];
    loss += stats.log_exp_sum - (logits->data[row, target] - stats.maximum);
  }

  loss = loss / static_cast<float>(rows);

  auto out = Tensor::create(Matrix(1, 1, loss), {logits}, "cross_entropy");
  Tensor *out_raw = out.get();
  out->backward_op = [logits, target_classes = std::move(target_classes),
                      probabilities = std::move(probabilities), out_raw] {
    for (size_t row{}; row < logits->data.rows(); row++) {
      for (std::size_t col = 0; col < probabilities.cols(); ++col) {
        float derivative = probabilities[row, col];
        if (col == target_classes[row]) {
          derivative -= 1.0f;
        }
        logits->grad[row, col] += derivative * out_raw->grad[0, 0] /
                                  static_cast<float>(probabilities.rows());
      }
    }
  };
  return out;
}

} // namespace aria
