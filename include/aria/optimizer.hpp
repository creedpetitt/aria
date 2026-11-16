#pragma once

#include <aria/tensor.hpp>
#include <aria/validate.hpp>

#include <unordered_set>
#include <utility>
#include <vector>

namespace aria {

class Optimizer {
public:
  virtual ~Optimizer() = default;

  Optimizer(std::vector<TensorPtr> parameters, float learning_rate)
      : parameters_(std::move(parameters)), learning_rate_(learning_rate) {
    validate::require_positive_finite(
      learning_rate_, "Learning rate must be positive and finite");
    for (const auto &parameter : parameters_) {
      validate::require_nonnull(parameter, "Optimizer parameters must not be null");
    }

    std::unordered_set<Tensor *> seen;
    seen.reserve(parameters_.size());
    std::erase_if(parameters_, [&](const TensorPtr &parameter) {
      return !seen.insert(parameter.get()).second;
    });
  }

  virtual void zero_grad() {
    for (const auto &parameter : parameters_) {
      parameter->grad.fill(0.0f);
    }
  }

  virtual void step() {
    for (const auto &parameter : parameters_) {
      parameter->data += parameter->grad * (-learning_rate_);
    }
  }

protected:
  std::vector<TensorPtr> parameters_;
  float learning_rate_;
};

} // namespace aria
