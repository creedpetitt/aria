# WebAssembly bindings

The optional `aria_wasm` target compiles the core with Emscripten and exposes
selected operations through Embind. The initial API includes Matrix construction,
`rows`, `cols`, `fill`, checked `get`/`set`, and returning `matmul(a, b)`.
Matrix multiplication uses the same tiled implementation as native C++.

Tensor, models, losses, optimizers, bulk typed-array transfer, and I/O are not
yet bound. This first module establishes compilation, JavaScript ownership,
exception propagation, and TypeScript declarations before exposing training.

## Build

Activate your Emscripten SDK so `emcc` and `emcmake` are on `PATH`. Node is also
needed by Emscripten to generate the TypeScript declarations. No vcpkg is needed.
Tested with Emscripten 7.0.0 and Node 20.20.2 for the smoke test.

The development presets enable Wasm alongside the native targets. From the
repository root, the normal commands build both:

```sh
cmake --preset release
cmake --build --preset release
```

In VS Code, select the `release` configure/build presets and build the default
`all` target. The editor process must have access to the activated SDK's PATH;
restart or launch VS Code from an SDK-enabled shell if `emcmake` is not found.
Building a specific native target such as `matrix_tests` only builds that target.
To update only the Wasm output, use `cmake --build build/release --target aria_wasm`.

The native build invokes a separate Emscripten configuration under its own
`wasm/` directory. For example, Release generates:

| File | Purpose |
| --- | --- |
| `build/release/wasm/aria.mjs` | ES-module loader and Embind interface |
| `build/release/wasm/aria.wasm` | Compiled engine and runtime |
| `build/release/wasm/aria.d.mts` | Generated TypeScript declarations for `aria.mjs` |

Debug uses `build/debug/wasm/`; I/O presets likewise keep their own Wasm build
under their native build directory. The child uses the parent's build type and
builds only the bindings, with I/O, tests, examples, and benchmarks disabled.
Native compiler and vcpkg toolchain settings are not forwarded to that child.

Every default build runs the child configure/build commands. The child's build
system tracks source and included-header dependencies and recompiles only when
needed. This is an incremental build, not a file watcher: save changes, then
build. Selecting a native compiler for the parent is expected and supported.

For a native-only development build, override the preset's setting:

```sh
cmake --preset release -DARIA_BUILD_WASM=OFF
cmake --build --preset release
```

Repeat the override when reconfiguring with that preset, or use a local preset
that sets it to `OFF`. Plain CMake configurations and library consumers still
default to `ARIA_BUILD_WASM=OFF`, so they do not require Emscripten.

You can also build only the module directly with Emscripten:

```sh
emcmake cmake -S . -B build/wasm -G Ninja \
  -DARIA_BUILD_WASM=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/wasm --target aria_wasm
```

This standalone route produces the same three files in `build/wasm/`. It uses
a separate compiler cache from the native build and does not spawn another
nested build.

The `.d.mts` extension lets TypeScript associate the declarations with the
`.mjs` import. Copy all three together when integrating into a frontend.
Serve the runtime files over HTTP(S), with `.wasm` served as `application/wasm`.
When using a bundler, ensure it preserves or copies the Wasm asset and resolves
it relative to the loader; the exact asset setup depends on the frontend build.

## JavaScript and TypeScript usage

The same code works in an ES module or a TypeScript module:

```ts
import createAriaModule from "./aria.mjs";

const aria = await createAriaModule();
const a = new aria.Matrix(2, 2);
const b = new aria.Matrix(2, 2);

try {
  a.set(0, 0, 1);
  a.set(0, 1, 2);
  a.set(1, 0, 3);
  a.set(1, 1, 4);
  b.set(0, 0, 5);
  b.set(0, 1, 6);
  b.set(1, 0, 7);
  b.set(1, 1, 8);

  const product = aria.matmul(a, b);
  try {
    console.log(product.get(0, 0)); // 19
    console.log(product.get(1, 1)); // 50
  } finally {
    product.delete();
  }
} finally {
  a.delete();
  b.delete();
}
```

Matrices initially contain zeros and store float32 values. Dimensions and
indices should be nonnegative integers. `get` and `set` use the engine's bounds
checks. Matmul requires `a.cols() === b.rows()` and returns an independently
owned Matrix; its inputs are preserved. `fill(value)` overwrites every element.

Call `.delete()` on each owned handle, including returned matrices. JavaScript
scope alone does not release an Embind object deterministically. Do not access
handles after deletion. Keep matrices within the module instance that created
them. The example uses `finally` so ordinary computation failures still release
its handles.

Bounds and shape failures propagate as exceptions that JavaScript can catch.
They are not currently translated into friendly JavaScript `Error` messages.
The module uses native Wasm exception handling, which must be supported by the
browser/runtime. No pthreads or shared-memory setup is needed.

Individual `get`/`set` calls are suitable for this small example. Bulk
`Float32Array` transfer is the next binding feature for real datasets.

## Worker integration

Initialize the module inside a module Web Worker and keep its Matrix handles
there. The frontend exchanges messages containing ordinary values and typed
arrays; Embind handles should not be sent between threads. Long computations
can then run without occupying the UI thread. This repository does not yet
provide a worker message protocol or a frontend.

For future training loops, process batches in chunks and yield between chunks
so the worker can handle progress and cancellation messages.

## Where the implementation lives

- `wasm/wasm.cpp` registers classes and functions. Its small `get`/`set` adapters
  use checked Matrix indexing; `select_overload` selects returning Matrix matmul.
- `cmake/AriaWasm.cmake` defines the optional build target and its flags.
- `include/aria/` remains the numerical implementation.

CMake uses `add_executable` to link the final Wasm module. `--no-entry` allows
this module to export bindings without a `main()` function. `-lembind` supplies
the binding runtime, modularized ES output supplies the asynchronous factory,
and memory growth lets the module expand its heap. Exception support is enabled
at both compile and link time. Declarations are generated from the bindings
with `--emit-tsd`, rather than maintained as a second handwritten API.

## Verification

After building, run the JavaScript smoke test from the repository root:

```sh
node tests/wasm_tests.mjs build/release/wasm/aria.mjs
```

For a different build directory (the default without an argument is the
standalone `build/wasm/aria.mjs`):

```sh
node tests/wasm_tests.mjs /path/to/aria.mjs
```

The test executes the actual Wasm module, checking results, checked access,
shape errors, continued operation after exceptions, explicit handle disposal,
and rectangular multiplication crossing the 64-element tile boundaries.
Node verification does not replace testing asset loading and workers in the
intended browser frontend.

See the official [Embind guide](https://emscripten.org/docs/porting/connecting_cpp_and_javascript/embind.html)
and [Emscripten CMake instructions](https://emscripten.org/docs/compiling/Building-Projects.html#cmake-based-projects)
for binding and toolchain details.
