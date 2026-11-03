# Aria Math

A C++23 machine learning library with a header-only core. Include `<aria.hpp>`
for the core, or an individual header such as `<aria/matrix.hpp>`. Public headers
live under `include/aria/` in namespace `aria`. The optional, separately compiled
I/O module reads text, JSON/JSONL, CSV, and PNG/JPEG files.

## Matrix operations

The following free functions live in namespace `aria`:

| Expression | Meaning | Output storage |
| --- | --- | --- |
| `a + b` | Elementwise addition | New result |
| `a * b` | Elementwise multiplication | New result |
| `a * scalar` | Scalar multiplication | New result |
| `matmul(a, b)` | Matrix multiplication | New result |
| `matmul(out, a, b)` | Matrix multiplication | Existing `out` buffer |
| `transpose(a)` | Transpose | New result |
| `transpose(out, a)` | Transpose | Existing `out` buffer |

Output-first overloads require an already correctly sized matrix. Matmul and
transpose require output storage separate from their inputs. Shape errors throw
`std::invalid_argument`. Both forms use the same numerical kernels.

```cpp
#include <aria.hpp>

aria::Matrix a{{1, 2}, {3, 4}};
aria::Matrix b{{5, 6}, {7, 8}};
auto product = aria::matmul(a, b);
aria::matmul(product, a, b);
auto transposed = aria::transpose(a);
a[0, 1] = 9.0f;
```

`matrix[row, col]` returns a reference, and checks bounds.
Const matrices return const references. `access_elem(row, col)` remains
available. Both throw `std::out_of_range` for invalid indices. Numerical kernels
can use direct buffer indexing after validating shapes once.

## Models and initialization

```cpp
aria::seed(69);
aria::Sequential network;
network.add<aria::Linear>(784, 32);
network.add<aria::ReLULayer>();
network.add<aria::Linear>(32, 10);
aria::Optimizer optimizer(network.parameters(), 0.03f);
```

## Tensors and training

Matrix operations calculate numbers. Tensor operations additionally record
parents and backward callbacks. Tensor `+` and `*` have the same arithmetic
meaning as Matrix `+` and `*`; Tensor `matmul(a, b)` performs matrix
multiplication and records its derivative rule. Output-first reuse overloads
currently apply to Matrix only. Named `transpose` currently applies to Matrix.

`tensor->data[row, col]` and `tensor->grad[row, col]` access stored floats;
those accesses do not create differentiable Tensor indexing operations.

Each input to Linear is currently `1 x input_features`. Its output is
`1 x output_features`. `combined_cross_entropy(scores, target_class)` accepts
one row of raw class scores and a zero-based correct class index, and returns
a `1 x 1` loss Tensor.

Training clears parameter gradients, calculates scores and loss, calls
`loss->backward()`, then calls `optimizer.step()`. Build a fresh forward graph
for each training example. `backward()` fills gradients; `step()` updates data.

## Build and test

```sh
cmake --preset release
cmake --build --preset release
ctest --preset release
```

Presets require CMake 3.21+ and Ninja. Use `debug` instead of `release` for a
debug build. Link CMake consumers to `aria::aria`, or add `include/` to your
compiler's include paths. The core requires no third-party dependencies or
separately compiled implementation.

## Optional I/O

I/O is off by default and is not included by `<aria.hpp>`. With `vcpkg` available
on `PATH`, run:

```sh
cmake --preset io-release
cmake --build --preset io-release
ctest --preset io-release
```

The presets locate vcpkg and load its toolchain automatically. No `VCPKG_ROOT`
export or machine-specific preset is needed. The manifest pins a registry
baseline and installs the enabled readers' dependencies. `io-debug` provides
the same readers in a debug build. The
`io-text` preset needs no vcpkg or external dependencies.

Consumers include `<aria/io.hpp>` and link `aria::io`. Parser headers and types
remain private to `src/io/`; linking `aria::aria` alone does not enable I/O.
Readers return matrices, strings, or small structs containing those values.
Creating differentiable tensors remains an explicit step:

```cpp
#include <aria.hpp>
#include <aria/io.hpp>

auto image = aria::io::load("digit.png", aria::io::Image{.grayscale = true});
auto input = aria::Tensor::create(std::move(image.pixels));
```

See [I/O formats, options, and examples](docs/io.md) for JSON schemas, CSV
column selection, image shapes, and individual reader switches.

See [the MNIST demo](examples/README.md) for dataset download and run instructions.
The demo prints only its final test accuracy. MNIST loading is demo code, separate
from the library API.
