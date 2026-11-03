#pragma once

#include <aria/layer.hpp>

#include <memory>
#include <utility>
#include <vector>

namespace aria {

class Sequential : public Layer {
public:
  template <typename LayerType, typename... Args> void add(Args &&...args) {
    layers_.push_back(std::make_unique<LayerType>(std::forward<Args>(args)...));
  }

  TensorPtr forward(const TensorPtr &input) const override {
    TensorPtr current = input;

    for (const auto &layer : layers_) {
      current = layer->forward(current);
    }

    return current;
  }

  std::vector<TensorPtr> parameters() const override {
    std::vector<TensorPtr> result;

    for (const auto &layer : layers_) {
      const auto parameters = layer->parameters();
      result.insert(result.end(), parameters.begin(), parameters.end());
    }

    return result;
  }

private:
  std::vector<std::unique_ptr<Layer>> layers_;
};

} // namespace aria
