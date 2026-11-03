#pragma once

#include <aria/tensor.hpp>

namespace aria {

class Layer {
public:
  virtual ~Layer() = default;

  virtual TensorPtr forward(const TensorPtr &input) const = 0;

  virtual std::vector<TensorPtr> parameters() const { return {}; }
};

} // namespace aria
