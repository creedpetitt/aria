#pragma once

#include <aria/init.hpp>
#include <aria/layer.hpp>
#include <aria/random.hpp>
#include <aria/tensor.hpp>
#include <aria/validate.hpp>

#include <cstddef>

namespace aria {

class Linear : public Layer {
public:
  Linear(std::size_t input_count, std::size_t output_count)
      : input_count_(input_count), output_count_(output_count),
        weight_(Tensor::create(
            init::xavier_uniform(input_count, output_count))),
        bias_(Tensor::create(Matrix(1, output_count))) {}

  TensorPtr forward(const TensorPtr &input) const override {
    validate::require_equal(input->data.cols(), input_count_,
                            "Linear input must have input_count columns");
    TensorPtr weighted_in = matmul(input, weight_);
    TensorPtr output = weighted_in + bias_;
    return output;
  }

  std::vector<TensorPtr> parameters() const override {
    return {weight_, bias_};
  }

private:
  std::size_t input_count_;
  std::size_t output_count_;
  TensorPtr weight_;
  TensorPtr bias_;
};

} // namespace aria
