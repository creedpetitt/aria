#pragma once

#include <aria/matrix.hpp>

#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace aria::io::detail {

[[noreturn]] inline void fail(const std::filesystem::path &path,
                               const std::string &message) {
  throw std::runtime_error("Aria I/O [" + path.string() + "]: " + message);
}

inline std::ifstream open_input(const std::filesystem::path &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    fail(path, "cannot open file");
  }
  return file;
}

inline std::size_t checked_product(std::size_t a, std::size_t b) {
  if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b) {
    throw std::runtime_error("dimensions overflow");
  }
  return a * b;
}

inline float as_float(double value) {
  if (!std::isfinite(value) ||
      std::abs(value) > std::numeric_limits<float>::max()) {
    throw std::runtime_error("expected a finite number in float range");
  }
  return static_cast<float>(value);
}

class MatrixBuilder {
public:
  MatrixBuilder() = default;
  explicit MatrixBuilder(std::size_t columns) : columns_(columns) {}

  void append(std::span<const float> row) {
    if (!columns_) {
      columns_ = row.size();
    }
    if (row.size() != *columns_) {
      throw std::runtime_error("inconsistent row width");
    }
    if (rows_ == std::numeric_limits<std::size_t>::max()) {
      throw std::runtime_error("too many rows");
    }
    checked_product(rows_ + 1, *columns_);
    if (!row.empty()) {
      values_.insert(values_.end(), row.begin(), row.end());
    }
    ++rows_;
  }

  Matrix finish() {
    return Matrix::from_buffer(rows_, columns_.value_or(0), std::move(values_));
  }

private:
  std::size_t rows_{};
  std::optional<std::size_t> columns_;
  std::vector<float> values_;
};

} // namespace aria::io::detail
