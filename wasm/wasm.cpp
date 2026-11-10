#include <aria.hpp>
#include <emscripten/bind.h>

using namespace emscripten;

EMSCRIPTEN_BINDINGS(aria_api) {
  class_<aria::Matrix>("Matrix")
      .constructor<std::size_t, std::size_t>()
      .function("rows", &aria::Matrix::rows)
      .function("cols", &aria::Matrix::cols)
      .function("fill", &aria::Matrix::fill)
      .function("get", optional_override(
          [](const aria::Matrix &m, std::size_t row, std::size_t col) {
            return m[row, col];
          }))
      .function("set", optional_override(
          [](aria::Matrix &m, std::size_t row, std::size_t col, float value) {
            m[row, col] = value;
          }));

  function("matmul",
           select_overload<aria::Matrix(const aria::Matrix &,
                                       const aria::Matrix &)>(&aria::matmul));
}

