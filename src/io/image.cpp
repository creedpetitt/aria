#include <aria/io.hpp>

#include "detail/common.hpp"

#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <memory>

namespace aria::io {

ImageData load(const std::filesystem::path &path, Image options) {
  int width = 0;
  int height = 0;
  int original_channels = 0;
  const int channels = options.grayscale ? 1 : 3;
  std::unique_ptr<unsigned char, decltype(&stbi_image_free)> pixels(
      stbi_load(path.string().c_str(), &width, &height, &original_channels, channels),
      stbi_image_free);
  if (!pixels) {
    const auto *reason = stbi_failure_reason();
    detail::fail(path, reason ? reason : "cannot decode image");
  }
  if (width <= 0 || height <= 0) {
    detail::fail(path, "invalid image dimensions");
  }
  try {
    const auto w = static_cast<std::size_t>(width);
    const auto h = static_cast<std::size_t>(height);
    const auto c = static_cast<std::size_t>(channels);
    const auto row_width = detail::checked_product(w, c);
    const auto count = detail::checked_product(h, row_width);
    std::vector<float> values;
    values.reserve(count);
    const float divisor = options.normalize ? 255.0f : 1.0f;
    for (std::size_t index = 0; index < count; ++index) {
      values.push_back(pixels.get()[index] / divisor);
    }
    return {Matrix::from_buffer(options.flatten ? 1 : h,
                                 options.flatten ? count : row_width,
                                 std::move(values)), w, h, c};
  } catch (const std::exception &error) {
    detail::fail(path, error.what());
  }
}

} // namespace aria::io
