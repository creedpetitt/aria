#include <aria/io.hpp>

#include "detail/common.hpp"

#include <nlohmann/json.hpp>

namespace aria::io {
namespace {

using Document = nlohmann::json;

const Document &select(const Document &record, const std::string &field) {
  if (field.empty()) {
    return record;
  }
  if (!record.is_object() || !record.contains(field)) {
    throw std::runtime_error("missing object field: " + field);
  }
  return record.at(field);
}

Document read_document(const std::filesystem::path &path) {
  auto file = detail::open_input(path);
  try {
    auto document = Document::parse(file);
    if (file.bad()) {
      detail::fail(path, "failed while reading JSON");
    }
    return document;
  } catch (const Document::exception &error) {
    detail::fail(path, error.what());
  }
}

std::vector<float> numeric_row(const Document &row) {
  if (!row.is_array()) {
    throw std::runtime_error("expected an array of numbers");
  }
  std::vector<float> values;
  values.reserve(row.size());
  for (const auto &value : row) {
    if (!value.is_number()) {
      throw std::runtime_error("expected a numeric array element");
    }
    values.push_back(detail::as_float(value.get<double>()));
  }
  return values;
}

std::string text_record(const Document &record, const std::string &field) {
  const auto &value = select(record, field);
  if (!value.is_string()) {
    throw std::runtime_error("expected a string in the selected text field");
  }
  return value.get<std::string>();
}

template <typename Function>
void for_each_record(const std::filesystem::path &path, std::size_t limit,
                     Function function) {
  auto file = detail::open_input(path);
  std::string line;
  std::size_t line_number = 0;
  std::size_t records = 0;
  while ((limit == 0 || records < limit) && std::getline(file, line)) {
    ++line_number;
    if (line.find_first_not_of(" \t\r") == std::string::npos) {
      continue;
    }
    try {
      function(Document::parse(line));
    } catch (const std::exception &error) {
      detail::fail(path, "JSONL line " + std::to_string(line_number) + ": " + error.what());
    }
    ++records;
  }
  if (file.bad()) {
    detail::fail(path, "failed while reading JSONL");
  }
}

} // namespace

Matrix load(const std::filesystem::path &path, Json options) {
  const auto document = read_document(path);
  try {
    const auto &rows = select(document, options.field);
    if (!rows.is_array()) {
      throw std::runtime_error("expected a numeric array or array of numeric rows");
    }
    detail::MatrixBuilder builder;
    if (!rows.empty() && !rows.front().is_array()) {
      builder.append(numeric_row(rows));
    } else {
      for (const auto &row : rows) {
        builder.append(numeric_row(row));
      }
    }
    return builder.finish();
  } catch (const std::exception &error) {
    detail::fail(path, error.what());
  }
}

Matrix load(const std::filesystem::path &path, Jsonl options) {
  detail::MatrixBuilder builder;
  for_each_record(path, options.max_records, [&](const Document &record) {
    builder.append(numeric_row(select(record, options.field)));
  });
  return builder.finish();
}

std::vector<std::string> load_text(const std::filesystem::path &path, Json options) {
  const auto document = read_document(path);
  try {
    std::vector<std::string> result;
    if (document.is_array()) {
      for (const auto &record : document) {
        result.push_back(text_record(record, options.field));
      }
    } else {
      const auto &value = select(document, options.field);
      if (value.is_array()) {
        for (const auto &record : value) {
          result.push_back(text_record(record, ""));
        }
      } else {
        result.push_back(text_record(value, ""));
      }
    }
    return result;
  } catch (const std::exception &error) {
    detail::fail(path, error.what());
  }
}

std::vector<std::string> load_text(const std::filesystem::path &path, Jsonl options) {
  std::vector<std::string> result;
  for_each_record(path, options.max_records, [&](const Document &record) {
    result.push_back(text_record(record, options.field));
  });
  return result;
}

} // namespace aria::io
