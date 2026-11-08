#pragma once

#include <aria/matrix.hpp>

#include <cstddef>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace aria::io {

enum class Format { text, json, jsonl, csv, image };
bool available(Format format) noexcept;

struct Text {
  bool lines{false};
  std::size_t max_records{0};
};

struct Json {
  std::string field;
};

struct Jsonl {
  std::string field;
  std::size_t max_records{0};
};

struct Csv {
  char delimiter{','};
  bool has_header{false};
  std::vector<std::size_t> columns;
  std::optional<std::size_t> target_column;
  std::size_t max_rows{0};
};

struct Image {
  bool grayscale{false};
  bool flatten{true};
  bool normalize{true};
};

struct Table {
  Matrix features;
  Matrix targets;
  std::vector<std::string> column_names;
};

struct ImageData {
  Matrix pixels;
  std::size_t width;
  std::size_t height;
  std::size_t channels;
};

std::vector<std::string> load_text(const std::filesystem::path &path, Text options = {});
std::vector<std::string> load_text(const std::filesystem::path &path, Json options);
std::vector<std::string> load_text(const std::filesystem::path &path, Jsonl options);

Matrix load(const std::filesystem::path &path, Json options);
Matrix load(const std::filesystem::path &path, Jsonl options);
Table load(const std::filesystem::path &path, Csv options);
ImageData load(const std::filesystem::path &path, Image options);

} // namespace aria::io
