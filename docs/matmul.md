# How matmul tiling works

`matmul(out, a, b)` in `include/aria/matrix.hpp` uses 64-element blocks along
all three dimensions. The returning `matmul(a, b)`, Tensor matmul, and Linear
layers all reach this implementation. Blocks are index ranges into the existing
buffers: no block matrices are allocated or copied. No restrict qualifiers or
new compiler flags are used.

## The calculation stays the same

For `A` shaped `M x K` and `B` shaped `K x N`, the output is `M x N`:

```text
C[i, j] = A[i, 0] * B[0, j]
        + A[i, 1] * B[1, j]
        + ...
        + A[i, K-1] * B[K-1, j]
```

Storage is row-major. `a_data[i * inner + k]` means `A[i, k]`, and
`b_data[k * cols + j]` means `B[k, j]`. The implementation checks shapes and
output aliasing first, then zeros the output once before accumulating products.

Previously, the loops were:

```cpp
for (std::size_t i = 0; i < rows; ++i) {
  for (std::size_t k = 0; k < inner; ++k) {
    const float a_val = a_data[i * inner + k];
    for (std::size_t j = 0; j < cols; ++j) {
      out_data[i * cols + j] += a_val * b_data[k * cols + j];
    }
  }
}
```

For a fixed output row `i`, this visits all of B. It then advances to the next
output row and visits all of B again. Traversing rows contiguously is already
good, but repeatedly revisiting a large B can make cache reuse less effective.

## The outer loops choose a smaller region

The current loop nest is:

```cpp
constexpr std::size_t block = 64;
for (std::size_t ii = 0; ii < rows; ii += block) {
  for (std::size_t kk = 0; kk < inner; kk += block) {
    for (std::size_t jj = 0; jj < cols; jj += block) {
      const std::size_t i_end = std::min(ii + block, rows);
      const std::size_t k_end = std::min(kk + block, inner);
      const std::size_t j_end = std::min(jj + block, cols);
      for (std::size_t i = ii; i < i_end; ++i) {
        for (std::size_t k = kk; k < k_end; ++k) {
          const float a_val = a_data[i * inner + k];
          for (std::size_t j = jj; j < j_end; ++j) {
            out_data[i * cols + j] += a_val * b_data[k * cols + j];
          }
        }
      }
    }
  }
}
```

There are two sets of coordinates:

| Coordinate | Meaning | Increment |
| --- | --- | --- |
| `ii` | Start of a group of output rows; also rows of A | 64 |
| `kk` | Start of a group along the shared dimension: columns of A, rows of B | 64 |
| `jj` | Start of a group of output columns; also columns of B | 64 |
| `i`, `k`, `j` | Individual coordinates within those groups | 1 |

The important change is in **all three inner loops' ranges**, not just the
three `std::min` calls. Previously, they covered the complete dimensions. Now
they visit only the region selected by `ii`, `kk`, and `jj`.

For one selection of the three outer coordinates, the inner loops perform:

```text
C[ii:i_end, jj:j_end] += A[ii:i_end, kk:k_end]
                     * B[kk:k_end, jj:j_end]
```

These slices describe ranges; they are not actual temporary objects. The inner
`j` loop still walks contiguous rows and the arithmetic statement is unchanged.

## What std::min does

Loop ends are exclusive, like ordinary C++ loops. For a dimension of 100 and a
block size of 64:

| Block start | Requested end | Clamped end | Indices visited |
| --- | --- | --- | --- |
| 0 | 64 | `min(64, 100) = 64` | 0 through 63 |
| 64 | 128 | `min(128, 100) = 100` | 64 through 99 |

Without the clamp, the second block would access indices 100 through 127,
which are outside the matrix. Each dimension has its own clamp because matrices
can be rectangular. A dimension smaller than 64 has one partial block; a
multiple of 64 has only full blocks.

The `std::min` calls handle edges. **They do not cause the speedup themselves.**
Their results are computed before the three innermost numerical loops.

## Walk through a 100 x 100 multiplication

The first outer selection is `ii=0, kk=0, jj=0`:

1. Update output rows 0–63 and columns 0–63 using shared indices 0–63.
2. For output row 0, use B's first 64 rows and first 64 columns.
3. For output row 1, immediately reuse that same section of B.
4. Continue through output row 63 before moving to another section of B.

Next, `jj=64` updates output rows 0–63 and columns 64–99, still using shared
indices 0–63. After both column blocks, `kk=64` supplies the remaining products
for shared indices 64–99. Those contributions are added to the existing output,
not substituted for it. Finally, `ii=64` handles output rows 64–99.

Each output element still receives every required product exactly once, with
`k` increasing from 0 to `K-1`. Different output elements are interleaved in a
different order. Operation count remains approximately `2 * M * K * N`.

## Why it can be faster

CPU caches hold recently accessed data. Reusing a smaller section of B across
many output rows makes it more likely that those reads are served by cache.
The output region is smaller too. The code expresses locality; it does not
explicitly pin memory in any cache.

A full `64 x 64` float tile is 16 KiB. Three such tiles total 48 KiB, but that
is only a rough working-set description: cache lines, other data, alignment,
and processor cache organization affect actual behavior. The chosen size is a
simple measured starting point, not a universal optimum or an L1-fit guarantee.

For tiny matrices, everything may already fit in cache, and extra loops can
cost more than they save. No threading is involved, and tiling does not reduce
the mathematical work.

## Native experiment

On an Intel Core i7-1165G7, using GCC 15.2, C++23, `-O3 -DNDEBUG`, one thread,
and the existing benchmark's allocation-inclusive timing:

| Square dimension | Original loops | 64-element tiling | Speedup |
| --- | --- | --- | --- |
| 100 | 0.12 ms | 0.10 ms | About 1.2x |
| 500 | 17.47 ms | 11.18 ms | About 1.56x |
| 1000 | 165.28 ms | 100.68 ms | About 1.64x |

These are medians of three rounds, each using the benchmark's one warmup and
three timed repetitions. Printed times are rounded, especially noticeably for
100 x 100. Variant order rotated between rounds. Restrict alone did not help
in that experiment; this implementation includes tiling only.

These measurements do not establish the same gains for rectangular neural-net
layers, complete training runs, other CPUs, or WebAssembly. Reproduce the current
benchmark with:

```sh
cmake --preset release
cmake --build --preset release --target matrix_bench
./build/release/matrix_bench 100 500 1000
```

Matrix reference tests cover both returning and output-buffer APIs, including
rectangular dimensions below, at, and above block boundaries. Existing tests
also cover empty dimensions, output reuse, and rejected output aliases.
