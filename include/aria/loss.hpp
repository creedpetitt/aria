#pragma once

#include <aria/autograd_ops.hpp>
#include <aria/tensor.hpp>
#include <aria/validate.hpp>

#include <cstddef>
#include <utility>
#include <vector>

namespace aria {

inline TensorPtr mean_squared_error(const TensorPtr &prediction,
                                    const Matrix &targets) {
  validate::require_nonnull(prediction, "MSE requires nonempty predictions");
  validate::require_nonempty(prediction->data, "MSE requires nonempty predictions");
  validate::require_shape(targets, prediction->data.rows(), prediction->data.cols());

  Matrix error = prediction->data - targets;
  float total = 0.0f;
  for (std::size_t row = 0; row < error.rows(); ++row) {
    for (std::size_t col = 0; col < error.cols(); ++col) {
      const float e = error[row, col];
      total += e * e;
    }
  }

  const float count = static_cast<float>(error.rows() * error.cols());
  auto out = Tensor::create(Matrix(1, 1, total / count),
                            {prediction}, AutogradOp::MSE);
  Tensor *out_raw = out.get();

  out->backward_op =
      [prediction, error = std::move(error), out_raw, count] {
        autograd::mse_backward(prediction->grad, error, out_raw->grad, count);
      };

  return out;
}

inline TensorPtr combined_cross_entropy(const TensorPtr &logits,
                                        std::vector<std::size_t> target_classes) {
  constexpr const char *input_error =
      "Cross entropy requires nonempty scores and one target per row";
  validate::require_nonnull(logits, input_error);
  validate::require_nonempty(logits->data, input_error);
  validate::require_equal(logits->data.rows(), target_classes.size(), input_error);
  validate::require_indices(target_classes, logits->data.cols(),
                            "Cross entropy target class is out of range");

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

  auto out = Tensor::create(Matrix(1, 1, loss), {logits}, AutogradOp::CrossEntropy);
  Tensor *out_raw = out.get();
  out->backward_op = [logits, target_classes = std::move(target_classes),
                      probabilities = std::move(probabilities), out_raw] {
    autograd::cross_entropy_backward(logits->grad, logits->data, probabilities,
                                     target_classes, out_raw->grad);
  };
  return out;
}

} // namespace aria
