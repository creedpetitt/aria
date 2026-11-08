#include <aria/io.hpp>

#include "detail/common.hpp"

#include <iterator>
#include <utility>

namespace aria::io {

std::vector<std::string> load_text(const std::filesystem::path &path, Text options) {
  auto file = detail::open_input(path);
  std::vector<std::string> records;
  if (options.lines) {
    std::string line;
    while ((options.max_records == 0 || records.size() < options.max_records) &&
           std::getline(file, line)) {
      if (!line.empty() && line.back() == '\r') {
        line.pop_back();
      }
      records.push_back(std::move(line));
    }
  } else {
    std::string text((std::istreambuf_iterator<char>(file)),
                     std::istreambuf_iterator<char>());
    if (!text.empty()) {
      records.push_back(std::move(text));
    }
  }
  if (file.bad()) {
    detail::fail(path, "failed while reading text");
  }
  return records;
}

} // namespace aria::io
