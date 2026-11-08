#include <aria/io.hpp>

#include "detail/common.hpp"

#include <csv.hpp>

#include <array>
#include <optional>

namespace aria::io {

Table load(const std::filesystem::path &path, Csv options) {
  auto file = detail::open_input(path);
  try {
    if (options.delimiter == '\0' || options.delimiter == '\n' ||
        options.delimiter == '\r' || options.delimiter == '"') {
      throw std::runtime_error("invalid CSV delimiter");
    }
    if (file.peek() == std::char_traits<char>::eof()) {
      return {};
    }
    csv::CSVFormat format;
    format.delimiter(options.delimiter);
    format.variable_columns(csv::VariableColumnPolicy::THROW);
    if (options.has_header) {
      format.header_row(0);
    } else {
      format.no_header();
    }
    csv::CSVReader reader(file, format);
    const auto names = options.has_header ? reader.get_col_names()
                                         : std::vector<std::string>{};
    std::vector<std::size_t> selected;
    std::vector<std::string> selected_names;
    std::optional<detail::MatrixBuilder> features;
    detail::MatrixBuilder targets(1);
    std::size_t source_width = 0;

    const auto configure = [&](std::size_t width) {
      source_width = width;
      if (options.target_column && *options.target_column >= width) {
        throw std::runtime_error("target column index out of range");
      }
      if (options.columns.empty()) {
        for (std::size_t col = 0; col < width; ++col) {
          if (!options.target_column || col != *options.target_column) {
            selected.push_back(col);
          }
        }
      } else {
        std::vector<bool> used(width);
        for (auto col : options.columns) {
          if (col >= width || used[col] ||
              (options.target_column && col == *options.target_column)) {
            throw std::runtime_error("invalid or duplicate feature column index");
          }
          used[col] = true;
          selected.push_back(col);
        }
      }
      if (options.has_header) {
        for (auto col : selected) {
          selected_names.push_back(names.at(col));
        }
      }
      features.emplace(selected.size());
    };
    if (options.has_header) {
      configure(names.size());
    }

    std::size_t record_number = 0;
    std::vector<float> values;
    for (auto &row : reader) {
      if (!features) {
        configure(row.size());
      }
      ++record_number;
      try {
        if (row.size() != source_width) {
          throw std::runtime_error("CSV column count mismatch");
        }
        values.clear();
        for (auto col : selected) {
          values.push_back(detail::as_float(row[col].get<double>()));
        }
        features->append(values);
        if (options.target_column) {
          const std::array target{
              detail::as_float(row[*options.target_column].get<double>())};
          targets.append(target);
        }
      } catch (const std::exception &error) {
        throw std::runtime_error("CSV record " + std::to_string(record_number) +
                                 ": " + error.what());
      }
      if (options.max_rows != 0 && record_number >= options.max_rows) {
        break;
      }
    }
    if (!features) {
      return {};
    }
    return {features->finish(), options.target_column ? targets.finish() : Matrix{},
            std::move(selected_names)};
  } catch (const std::exception &error) {
    detail::fail(path, error.what());
  }
}

} // namespace aria::io
