#pragma once

#include <aria/layer.hpp>

namespace aria {

class ReLULayer : public Layer {
public:
  TensorPtr forward(const TensorPtr &input) const override {
    return aria::relu(input);
  }
};

} // namespace aria
