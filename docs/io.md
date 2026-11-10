# Optional data readers

`aria::io` gets file contents into ordinary strings or row-major float matrices.
It does not tokenize text, create an autograd graph, choose model dimensions,
or run training. Include `<aria/io.hpp>` and link the CMake target `aria::io`.
The header-only core remains independent of this target.

## Build

With CMake 3.21+, Ninja, and `vcpkg` available on `PATH`:

```sh
cmake --preset io-release
cmake --build --preset io-release
ctest --preset io-release
```

The presets use `cmake/toolchains/Vcpkg.cmake` to find the vcpkg executable on
`PATH`, resolve symlinks, and include its toolchain. No environment export or
machine-specific path is required. Configuration installs dependencies
automatically using `vcpkg.json`. No global package installation is needed.
`io-debug` uses Debug instead of Release;
`release` and `debug` build only the core. `io-text` enables only plain text
and needs no vcpkg installation.

| CMake option | Default | Reader dependency |
| --- | --- | --- |
| `ARIA_BUILD_IO` | `OFF` | Enables the compiled I/O target |
| `ARIA_IO_JSON` | `ON` | `nlohmann-json` for JSON and JSONL |
| `ARIA_IO_CSV` | `ON` | `vincentlaucsb-csv-parser` |
| `ARIA_IO_IMAGE` | `ON` | `stb` for PNG/JPEG |

Reader switches take effect only when `ARIA_BUILD_IO=ON`. For example, to
exclude images from a full I/O build:

```sh
cmake --preset io-release -DARIA_IO_IMAGE=OFF
cmake --build --preset io-release
```

Set the override again when reconfiguring with that preset, since `io-release`
explicitly enables all three readers. For a persistent personal configuration,
create a local preset in the ignored `CMakeUserPresets.json`.

Dependencies are found in small functions under `cmake/dependencies/`. Their
includes are private to the compiled reader implementations. The CSV package
also brings its own support dependencies through vcpkg. A static-library
consumer still links the required CSV library through CMake; private headers
do not mean the implementation has no link dependencies.

Check reader availability with `aria::io::available(aria::io::Format::json)`.
Calling a reader that was disabled throws an error naming its CMake switch.

## Plain text

```cpp
auto documents = aria::io::load_text("book.txt");
auto lines = aria::io::load_text("lines.txt", aria::io::Text{
    .lines = true, .max_records = 100});
```

The default returns the entire file as one string. An empty file returns an
empty vector. Line mode removes line endings, handles CRLF, and keeps blank
lines. `max_records=0` means no limit. Text bytes are preserved without encoding
conversion, UTF-8 validation, or tokenization. Text must be encoded into numbers
before a model can consume it.

## Numerical JSON and JSONL

```cpp
auto matrix = aria::io::load("numbers.json", aria::io::Json{});
auto features = aria::io::load("dataset.json", aria::io::Json{.field = "features"});
auto rows = aria::io::load("dataset.jsonl", aria::io::Jsonl{
    .field = "features", .max_records = 1000});
```

JSON accepts a flat numeric array as one row, or equally sized arrays as rows:

```json
[[1, 2, 3], [4, 5, 6]]
```

That input produces a `2 x 3` Matrix. With `.field="features"`, the JSON file
instead contains an object such as `{"features": [[1, 2], [3, 4]]}`. A field is
one direct object key, not a nested path.

Numerical JSONL accepts one numeric array per nonblank line, or one object per
line containing the selected array:

```json
{"features": [1, 2, 3]}
{"features": [4, 5, 6]}
```

Each record becomes one matrix row. JSONL is parsed line by line; results still
accumulate in memory. `max_records=0` reads all records. Blank lines are skipped.
Input beyond the requested record limit is not validated. Ragged rows, strings
inside numeric arrays, and nonfinite or out-of-range float values are rejected.
Empty JSON `[]` produces `0 x 0`; `[[], []]` produces `2 x 0`.

## Text JSON and JSONL

```cpp
auto messages = aria::io::load_text("messages.json", aria::io::Json{.field = "text"});
auto records = aria::io::load_text("messages.jsonl", aria::io::Jsonl{.field = "text"});
```

Text JSON accepts a string, an array of strings, an array of objects with a
selected string field, or an object whose selected field contains a string or
array of strings. For example:

```json
[{"text": "first message"}, {"text": "second message"}]
```

Text JSONL accepts one string or one object with a selected string field per
nonblank line. These functions return `std::vector<std::string>`, not tensors.

## CSV

```cpp
auto table = aria::io::load("samples.csv", aria::io::Csv{
    .has_header = true,
    .columns = {0, 1},
    .target_column = 2});
// table.features: N x 2; table.targets: N x 1
```

Column indices are zero-based. An empty `columns` list selects all columns
except the target. Selected columns and the optional target must be numeric;
unselected columns can contain text. Quoted fields, quoted delimiters, and
multiline quoted fields are handled by the CSV parser. Rows must have consistent
source widths, including unselected columns.

`delimiter` defaults to comma, `has_header` defaults to false, and `max_rows=0`
reads all rows. `column_names` contains the selected feature names when a header
is present. Without a target, `targets` is an empty `0 x 0` matrix. Target values
are floats; interpreting them as class indices or regression values is the
caller's responsibility. No categorical encoding or normalization is applied.

## Images

```cpp
auto image = aria::io::load("digit.png", aria::io::Image{.grayscale = true});
auto rgb = aria::io::load("photo.jpg", aria::io::Image{.flatten = false});
```

`ImageData` contains `pixels`, `width`, `height`, and the returned `channels`
count. Images are decoded into one grayscale channel or three interleaved RGB
channels; alpha is discarded. By default, byte values are divided by 255 and
the matrix is flattened to `1 x (height * width * channels)`. With
`flatten=false`, its shape is `height x (width * channels)`. With
`normalize=false`, float values remain in the range 0–255.

Only PNG and JPEG decoding is enabled. Resizing, directory discovery, label
assignment, and batching belong to the calling dataset code for now.

## Storage, errors, and tests

Readers return owned data. Numerical builders transfer a float vector into
`Matrix::from_buffer` after checking dimensions, avoiding a second full buffer
copy. This factory also works for callers with an existing float vector:

```cpp
auto matrix = aria::Matrix::from_buffer(rows, cols, std::move(values));
```

Pass an rvalue to transfer storage; passing an lvalue copies it. Numerical
conversion to float can round values. Readers throw `std::runtime_error` for
format and file errors, with the path and, when available, the record or line.
Allocation failures can propagate as standard allocation exceptions.

Tests generate text, JSON/JSONL, CSV, PNG, and JPEG fixtures locally. No dataset
download is required. Tests cover shapes and values, disabled readers, bad
input, column selection, and wrapping loaded data in a differentiable Tensor.
