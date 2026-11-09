#pragma once

#include <aria/matrix.hpp>
#include <aria/random.hpp>

#include <cmath>
#include <cstddef>
#include <random>
#include <stdexcept>

namespace aria::init {

inline Matrix xavier_uniform(std::size_t input_count, std::size_t output_count,
                             std::mt19937 &generator = aria::detail::default_generator()) {
  if (input_count == 0 || output_count == 0) {
    throw std::invalid_argument("Initialization dimensions must be positive");
  }

  const float limit = std::sqrt(6.0f / (static_cast<float>(input_count) +
                                        static_cast<float>(output_count)));
  std::uniform_real_distribution<float> distribution(-limit, limit);
  Matrix weights(input_count, output_count);
  for (std::size_t row = 0; row < input_count; ++row) {
    for (std::size_t col = 0; col < output_count; ++col) {
      weights[row, col] = distribution(generator);
    }
  }
  return weights;
}

} // namespace aria::init
