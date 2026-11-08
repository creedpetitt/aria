#include <aria/io.hpp>

#include "detail/common.hpp"

namespace aria::io {

bool available(Format format) noexcept {
  switch (format) {
  case Format::text: return true;
  case Format::json:
  case Format::jsonl: return ARIA_IO_HAS_JSON;
  case Format::csv: return ARIA_IO_HAS_CSV;
  case Format::image: return ARIA_IO_HAS_IMAGE;
  }
  return false;
}

#if !ARIA_IO_HAS_JSON
Matrix load(const std::filesystem::path &path, Json) {
  detail::fail(path, "JSON support was not built (enable ARIA_IO_JSON)");
}
Matrix load(const std::filesystem::path &path, Jsonl) {
  detail::fail(path, "JSONL support was not built (enable ARIA_IO_JSON)");
}
std::vector<std::string> load_text(const std::filesystem::path &path, Json) {
  detail::fail(path, "JSON support was not built (enable ARIA_IO_JSON)");
}
std::vector<std::string> load_text(const std::filesystem::path &path, Jsonl) {
  detail::fail(path, "JSONL support was not built (enable ARIA_IO_JSON)");
}
#endif

#if !ARIA_IO_HAS_CSV
Table load(const std::filesystem::path &path, Csv) {
  detail::fail(path, "CSV support was not built (enable ARIA_IO_CSV)");
}
#endif

#if !ARIA_IO_HAS_IMAGE
ImageData load(const std::filesystem::path &path, Image) {
  detail::fail(path, "Image support was not built (enable ARIA_IO_IMAGE)");
}
#endif

} // namespace aria::io
