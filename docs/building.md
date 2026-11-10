# Build and integration

Aria provides two library targets:

- `aria::aria`: header-only core. CMake passes its include directory and C++23
  requirement to your target. There is no core `.a` or `.so` file.
- `aria::io`: optional compiled static library. It also links the core, so an I/O
  consumer only needs to link `aria::io`.

CMake configures builds and target relationships. vcpkg supplies third-party
packages; it is optional for the core and text-only I/O.

## Supported combinations

| Use | Include in C++ | Build integration | External dependencies |
| --- | --- | --- | --- |
| Core with CMake | `<aria.hpp>` or individual core headers | Link `aria::aria` | None; no vcpkg needed |
| Core without CMake | Same headers | Add `include/` to the compiler search path and use C++23 | None |
| Text-only I/O | `<aria/io.hpp>` plus any core headers used | Enable I/O, disable JSON/CSV/images, link `aria::io` | None |
| Full or selected I/O with vcpkg | Same headers | Enable readers, configure with the vcpkg toolchain, link `aria::io` | vcpkg supplies enabled readers' packages |
| Full or selected I/O without vcpkg | Same headers | Enable readers, provide their CMake packages, link `aria::io` | Caller supplies enabled readers' dependencies |

Headers declare the names used by your C++ source. CMake targets configure how
that source builds. With CMake, include the headers **and** link the appropriate
target. For the header-only core this adds build settings, not a binary library.

`<aria.hpp>` includes other core headers. It is an umbrella header, not a
standalone amalgamated file: retain the whole `include/` tree. It does not
include I/O. I/O requires compiled source and cannot be used just by including
`<aria/io.hpp>`.

## Core in your CMake project

Keep the repository as a checkout or submodule and add it as a subdirectory:

```cmake
cmake_minimum_required(VERSION 3.21)
project(my_model LANGUAGES CXX)

add_subdirectory(path/to/aria aria-build)
add_executable(my_model main.cpp)
target_link_libraries(my_model PRIVATE aria::aria)
```

Your `main.cpp` can now include `<aria.hpp>`. No vcpkg toolchain is needed.
The repository currently has no install/export rules or published vcpkg port;
`find_package(aria)` and installing Aria through vcpkg are not provided.

Without CMake, the equivalent core-only build is:

```sh
c++ -std=c++23 -I/path/to/aria/include main.cpp -o my_model
```

## Optional I/O in your CMake project

For dependency-free text readers, set the options before `add_subdirectory`:

```cmake
set(ARIA_BUILD_IO ON CACHE BOOL "" FORCE)
set(ARIA_IO_JSON OFF CACHE BOOL "" FORCE)
set(ARIA_IO_CSV OFF CACHE BOOL "" FORCE)
set(ARIA_IO_IMAGE OFF CACHE BOOL "" FORCE)
add_subdirectory(path/to/aria aria-build)

add_executable(my_model main.cpp)
target_link_libraries(my_model PRIVATE aria::io)
```

For JSON, CSV, or image readers, enable their switches and provide dependencies.
The repository's `io-release` and `io-debug` presets use vcpkg automatically.
Those presets are for building this repository. When embedding Aria, configure
**your parent project** with a vcpkg toolchain and declare the needed dependencies
in its manifest. Aria's subdirectory manifest is not automatically installed by
the parent's manifest build.

Without vcpkg, the current reader configuration expects:

| Reader | CMake dependency |
| --- | --- |
| JSON/JSONL | `nlohmann_json` config package, version 3.11 or newer |
| CSV | `unofficial-vincentlaucsb-csv-parser` config package |
| PNG/JPEG | `Stb`, supplying `Stb_INCLUDE_DIR` |

Use `CMAKE_PREFIX_PATH` for package locations, or provide a suitable CMake find
module where needed. The CSV package name and Stb discovery follow vcpkg's
conventions; arbitrary system installations may need an adapter. Disabling a
reader removes its dependency. See [the I/O guide](io.md) for reader details.

You could compile I/O manually with another build system, but you would need to
handle its source files, reader definitions, dependencies, and linking yourself.
CMake provides that wiring.

## Building Aria itself

A fresh plain configuration enables only the core by default:

```sh
cmake -S . -B build/library
cmake --build build/library
```

The core is header-only, so there is nothing to compile in this configuration.
Enable `ARIA_BUILD_IO` to build the compiled I/O library. Optional executables
are controlled separately:

| Option | Plain CMake default | Purpose |
| --- | --- | --- |
| `ARIA_BUILD_IO` | `OFF` | Compiled data readers |
| `ARIA_BUILD_WASM` | `OFF` | JavaScript/Wasm module; requires Emscripten |
| `ARIA_BUILD_EXAMPLES` | `OFF` | MNIST and enabled tabular demos |
| `ARIA_BUILD_BENCHMARKS` | `OFF` | Matrix benchmark |
| `BUILD_TESTING` | `OFF` | Core and enabled I/O tests |

The repository's development presets explicitly enable tests, examples, and
benchmarks, plus Wasm bindings. Activate the Emscripten SDK on `PATH` when
using these presets, or configure with `-DARIA_BUILD_WASM=OFF` for native-only work. Dataset training
tests are registered only when examples are enabled and their data is present.
Cached CMake options persist when reusing an existing build directory.

For browser or TypeScript integration, see [the WebAssembly guide](wasm.md).
Native builds with `ARIA_BUILD_WASM=ON` automatically run a separate Emscripten
build under their build directory. A direct `emcmake` build is also supported.
