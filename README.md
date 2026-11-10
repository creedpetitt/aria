# Aria Math

A small C++23 machine learning engine with a header-only core: float matrices,
autograd tensors, Linear and ReLU layers, Sequential models, SGD, and losses
for regression and classification. Training uses explicit mini-batch loops.

Include `<aria.hpp>` for the core, or an individual header such as
`<aria/matrix.hpp>`. Core types and functions live in namespace `aria`.
The core has no third-party dependencies or separately compiled implementation.
An optional I/O module reads text, JSON/JSONL, CSV, and PNG/JPEG files.

## Basic usage

Matrix operations work directly on numbers:

```cpp
#include <aria.hpp>

aria::Matrix a{{1, 2}, {3, 4}};
aria::Matrix b{{5, 6}, {7, 8}};
auto product = aria::matmul(a, b);
auto transposed = aria::transpose(a);
a[0, 1] = 9.0f;
```

Tensor operations record a graph for backward propagation. A model and one
training step look like this:

```cpp
#include <aria.hpp>

int main() {
  aria::seed(69);
  aria::Sequential network;
  network.add<aria::Linear>(784, 32);
  network.add<aria::ReLULayer>();
  network.add<aria::Linear>(32, 10);
  aria::Optimizer optimizer(network.parameters(), 0.03f);

  // Two placeholder images; replace with your batch of flattened pixels.
  aria::Matrix batch(2, 784);
  optimizer.zero_grad();
  auto scores = network.forward(aria::Tensor::create(batch));
  auto loss = aria::combined_cross_entropy(scores, {0, 1});
  loss->backward();
  optimizer.step();
}
```

Rows are examples and columns are features. Build a fresh forward graph for each
batch. For regression, use an output layer with the desired number of values
and `mean_squared_error(predictions, targets)` with a same-shaped target Matrix.

See [the public API reference](docs/public-api.md) for operations, shapes,
output-buffer overloads, gradient behavior, and initialization.

## Use in a CMake project

Add this repository as a subdirectory and link `aria::aria`:

```cmake
cmake_minimum_required(VERSION 3.21)
project(my_model LANGUAGES CXX)

add_subdirectory(path/to/aria aria-build)
add_executable(my_model main.cpp)
target_link_libraries(my_model PRIVATE aria::aria)
```

The target supplies the include path and C++23 requirement. You can also use the
headers directly by adding `include/` to your compiler's include paths and
compiling as C++23. `<aria.hpp>` is an umbrella header: keep the whole
`include/` directory, not just that file.

Plain CMake builds expose the library targets without building executables.
See [build and integration options](docs/building.md) for core-only, I/O,
vcpkg, and direct-header usage.

## Build and test this repository

With CMake 3.21+, Ninja, a compiler supporting C++23, and the Emscripten SDK
activated on `PATH`:

```sh
cmake --preset release
cmake --build --preset release
ctest --preset release
```

These development presets enable tests, examples, benchmarks, and Wasm bindings.
The same build updates the Wasm output in `build/release/wasm/` automatically.
Plain CMake configuration leaves these options off. Use `debug` instead of
`release` for a debug build. For native-only development, configure with
`cmake --preset release -DARIA_BUILD_WASM=OFF`.

## WebAssembly

An optional Emscripten build exposes Matrix construction, element access, and
matrix multiplication to JavaScript and TypeScript. It generates an ES module,
a `.wasm` binary, and TypeScript declarations. See [the Wasm guide](docs/wasm.md)
for build commands, usage, and ownership. Training bindings are not exposed yet.

## Optional I/O

I/O is off by default and is not included by `<aria.hpp>`. With `vcpkg` available
on `PATH`, build all readers with:

```sh
cmake --preset io-release
cmake --build --preset io-release
ctest --preset io-release
```

The presets find vcpkg and load its toolchain automatically. `io-debug` provides
the same readers in a debug build; `io-text` needs no external dependencies.
Consumers include `<aria/io.hpp>` and link `aria::io`.

```cpp
#include <aria.hpp>
#include <aria/io.hpp>

auto image = aria::io::load("digit.png", aria::io::Image{.grayscale = true});
auto input = aria::Tensor::create(std::move(image.pixels));
```

See [the I/O guide](docs/io.md) for reader options, dependencies, and data formats.
See [the examples guide](examples/README.md) for housing regression, Titanic
classification, and MNIST download and run instructions.

## Project layout

| Path | Contents |
| --- | --- |
| `include/aria.hpp`, `include/aria/` | Public core headers and optional I/O declarations |
| `src/io/` | Compiled data readers |
| `wasm/` | Optional Emscripten bindings; [usage guide](docs/wasm.md) |
| `docs/` | [Public API](docs/public-api.md), [build guide](docs/building.md), and [I/O guide](docs/io.md) |
| `examples/` | Tabular training examples and batching helpers |
| `main.cpp`, `mnist.hpp` | MNIST demo and dataset helper |
| `tests/` | Matrix, autograd, and data-reader tests |
| `benchmarks/` | Matrix benchmarks; [matmul tiling explained](docs/matmul.md) |
| `cmake/` | Build options, dependencies, and toolchains |

Dataset preparation and batching helpers remain example code, separate from
the library API.
