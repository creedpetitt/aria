#pragma once

#include <cstdint>
#include <random>

namespace aria {

namespace detail {

inline std::mt19937 &default_generator() {
  static std::mt19937 generator;
  return generator;
}

} // namespace detail

inline void seed(std::uint32_t value) {
  detail::default_generator().seed(value);
}

} // namespace aria
