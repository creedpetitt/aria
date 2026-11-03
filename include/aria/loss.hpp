#pragma once

#include <aria/tensor.hpp>

#include <cstddef>
#include <stdexcept>
#include <utility>

namespace aria {

inline TensorPtr squared_error(const TensorPtr &prediction, float target) {
  auto error = prediction + Tensor::create(Matrix(1, 1, -target));
  return error * error;
}

inline TensorPtr combined_cross_entropy(const TensorPtr &logits,
                                        std::size_t target_class) {
  if (!logits || logits->data.rows() != 1 ||
      target_class >= logits->data.cols()) {
    throw std::invalid_argument(
        "Cross entropy requires one row of scores and a valid target class");
  }

  Matrix probabilities(1, logits->data.cols());
  const auto stats = detail::softmax_row(probabilities, logits->data, 0);
  const float loss_value =
      stats.log_exp_sum -
      (logits->data.access_elem(0, target_class) - stats.maximum);

  auto out =
      Tensor::create(Matrix(1, 1, loss_value), {logits}, "cross_entropy");
  Tensor *out_raw = out.get();
  out->backward_op = [logits, target_class,
                      probabilities = std::move(probabilities), out_raw] {
    for (std::size_t col = 0; col < probabilities.cols(); ++col) {
      float derivative = probabilities.access_elem(0, col);
      if (col == target_class) {
        derivative -= 1.0f;
      }
      logits->grad.access_elem(0, col) +=
          derivative * out_raw->grad.access_elem(0, 0);
    }
  };
  return out;
}

} // namespace aria
