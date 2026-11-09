#pragma once

#include <aria/tensor.hpp>

#include <cmath>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

namespace aria {

// plain gradient descent optimizer
class Optimizer {
public:
  Optimizer(std::vector<TensorPtr> parameters, float learning_rate)
      : parameters_(std::move(parameters)), learning_rate_(learning_rate) {
    if (!std::isfinite(learning_rate_) || learning_rate_ <= 0.0f) {
      throw std::invalid_argument("Learning rate must be positive and finite");
    }
    for (const auto &parameter : parameters_) {
      if (!parameter) {
        throw std::invalid_argument("Optimizer parameters must not be null");
      }
    }

    std::unordered_set<Tensor *> seen;
    seen.reserve(parameters_.size());
    std::erase_if(parameters_, [&](const TensorPtr &parameter) {
      return !seen.insert(parameter.get()).second;
    });
  }

  void zero_grad() {
    for (const auto &parameter : parameters_) {
      parameter->grad.fill(0.0f);
    }
  }

  void step() {
    for (const auto &parameter : parameters_) {
      parameter->data += parameter->grad * (-learning_rate_);
    }
  }

private:
  std::vector<TensorPtr> parameters_;
  float learning_rate_;
};

} // namespace aria
