# Public API

Include `<aria.hpp>` for the header-only core, or an individual header such as
`<aria/matrix.hpp>`. Core types and functions live in namespace `aria`;
initializers live in `aria::init`. Names in `aria::detail` are internal helpers.
For optional data readers, see [the I/O reference](io.md).

## Matrix construction and storage

`Matrix` owns contiguous row-major `float` values. It supports copying and
moving; copies own independent storage.

| API | Purpose |
| --- | --- |
| `Matrix()` | Empty matrix |
| `Matrix(rows, cols, value = 0.0f)` | Allocate and fill a matrix |
| `Matrix{{1, 2}, {3, 4}}` | Construct from equally sized rows |
| `Matrix::from_buffer(rows, cols, values)` | Take a `std::vector<float>` by value; move it to transfer storage |
| `rows()`, `cols()` | Dimensions |
| `fill(value)` | Overwrite every element |
| `swap(other)` | Exchange dimensions and storage |
| `stream << matrix` | Print values |

`from_buffer` requires exactly `rows * cols` values. Invalid buffer sizes or
unequal initializer-list row lengths throw `std::invalid_argument`.

## Matrix operations

The following free functions live in namespace `aria`:

| Expression | Meaning |
| --- | --- |
| `a + b` | Elementwise addition |
| `a - b` | Elementwise subtraction |
| `a * b` | Elementwise multiplication |
| `a * scalar` | Scalar multiplication |
| `matmul(a, b)` | Matrix multiplication |
| `transpose(a)` | Transpose |
| `softmax(a)` | Softmax over each row |

These operations return a new Matrix and preserve their inputs. Each also has an
output-buffer form that shares the same numerical implementation:

| Output-buffer form | Meaning |
| --- | --- |
| `add(out, a, b)` | Elementwise addition |
| `subtract(out, a, b)` | Elementwise subtraction |
| `multiply(out, a, b)` | Elementwise multiplication |
| `scale(out, a, scalar)` | Scalar multiplication |
| `matmul(out, a, b)` | Matrix multiplication |
| `transpose(out, a)` | Transpose |
| `softmax(out, a)` | Softmax over each row |

The output-buffer forms overwrite `out` without allocating or resizing on
success. It must already have the exact result shape. For elementwise operations,
the output may be either input if that input already has the result shape,
including `scale(a, a, scalar)`. Softmax also supports `softmax(a, a)`.
Matmul and transpose require output separate from their inputs.
Shape errors throw `std::invalid_argument` before writing any elements. Addition
and subtraction broadcast rows and columns when their sizes match or one is 1.
Elementwise multiplication requires equal input shapes. Compound assignments
`a += b`, `a -= b`, `a *= b`, and `a *= scalar` use the same output-buffer
functions, preserve `a`'s dimensions, and return a reference to `a`.

The ordinary returning forms call the same output-buffer implementations.
For `matmul`, inputs `M x K` and `K x N` produce `M x N`; transpose swaps
rows and columns. Softmax normalizes each row and requires at least one column.

Matmul uses 64-element cache tiling in the shared Matrix implementation, so
Tensor matmul and Linear layers use it too. See [how matmul tiling works](matmul.md)
for the loop order, boundary handling, and measured native performance.

```cpp
#include <aria.hpp>

aria::Matrix a{{1, 2}, {3, 4}};
aria::Matrix b{{5, 6}, {7, 8}};
auto product = aria::matmul(a, b);
auto transposed = aria::transpose(a);
a[0, 1] = 9.0f;

aria::Matrix work(2, 2);
aria::add(work, a, b);
aria::scale(work, work, 0.5f); // Reuses the same buffer.
work += b;
aria::matmul(product, a, b); // Overwrites the existing product buffer.
aria::transpose(transposed, a);
aria::softmax(work, work);
```

Matrices expose two access interfaces:

- `matrix[row, col]` checks row and column bounds and throws `std::out_of_range`
  for invalid indices. It returns `float&`, or `const float&` for a const matrix.
- `data()` exposes unchecked contiguous row-major storage as `float*`, or
  `const float*` for a const matrix.

Tensor operations, losses, initialization, and individual element access in
examples and reference tests use `matrix[row, col]`. Direct buffer access for
numerical computation is confined to the Matrix implementations in `matrix.hpp`.
Those kernels validate required shapes before their loops and cache buffer
pointers and dimensions. A newly constructed output already has the dimensions
supplied to its constructor. Tensor backward retains its gradient shape checks.

| Internal calculation | Access pattern |
| --- | --- |
| Matrix arithmetic | Direct buffers; flat loops for equal-shaped elementwise work, row-major indexing for broadcasting, matmul, and transpose |
| Softmax helper in `matrix.hpp` | Direct row buffers, shared by Matrix softmax and cross-entropy |
| Tensor addition backward | Checked row/column indexing to reduce broadcast axes |
| ReLU forward/backward and initialization | Checked row/column loops |
| Cross-entropy | Checked indexing, saved probability dimensions, and a cached incoming scalar gradient |
| Tensor arithmetic composition | Existing Matrix operators and functions; `+=` accumulates gradients |
| Optimizer updates | `data += grad * (-learning_rate)` reuses parameter storage; the scaled-gradient temporary remains |

## Tensors and training

Matrix operations calculate numbers. Tensor operations additionally record
parents and backward callbacks. Tensor `+` and `*` have the same arithmetic
meaning as Matrix `+` and `*`; broadcast addition sums gradients back into each
input's original shape. Tensor `matmul(a, b)` performs matrix
multiplication and records its derivative rule. Named `transpose` and `softmax`
currently apply to Matrix.

`tensor->data[row, col]` and `tensor->grad[row, col]` access stored floats;
those accesses do not create differentiable Tensor indexing operations.

Each input to Linear is `batch_rows x input_features`. Its output is
`batch_rows x output_features`; every row shares the layer's weights and bias.
`combined_cross_entropy(scores, target_classes)` accepts a nonempty matrix of
raw class scores and a `std::vector<std::size_t>` with one zero-based target per
row. It returns the mean loss as a `1 x 1` Tensor. A one-row call uses `{target}`.

`mean_squared_error(predictions, targets)` accepts a prediction Tensor and a
same-shaped, nonempty Matrix of regression targets. It returns a `1 x 1` Tensor
averaging squared errors over every row and output column.

Training clears parameter gradients, calculates scores and loss, calls
`loss->backward()`, then calls `optimizer.step()`. Build a fresh forward graph
for each training batch. `backward()` fills gradients; `step()` updates data.

### Tensor lifetime and gradients

`TensorPtr` is `std::shared_ptr<Tensor>`. Use `Tensor::create(matrix)` to create
nodes, and copy or move the pointer to share or transfer a handle. Tensor nodes
themselves cannot be copied or moved because backward callbacks retain their
addresses. Passing a Matrix by move transfers its storage into the node.

A Tensor exposes `data`, `grad`, an operation label `op`, parent nodes `prev`,
and `backward_op`. The latter three describe the graph. The factory also accepts
an operation label, or a vector of parents and an optional operation label,
for manually constructed operations.

`backward()` traverses the reachable graph in reverse topological order and
seeds the output gradient with ones. For a nonscalar output, this differentiates
the sum of its elements. Input gradients accumulate; it does not clear them.
Use a fresh forward graph per batch and clear parameter gradients with the
optimizer before each backward pass. Finish backward before updating parameter
data or changing values used by the graph.

Tensor operations currently include `+`, elementwise `*`, `matmul`, and `relu`.
Multiplication requires matching shapes; addition supports Matrix broadcasting
and reduces broadcast gradients to the original input shapes. There are no
Tensor output-buffer overloads: these operations create new graph nodes.

## Layers and initialization

| API | Behavior |
| --- | --- |
| `Layer::forward(input) const` | Virtual operation returning a `TensorPtr` |
| `Layer::parameters() const` | Virtual parameter list; empty by default |
| `Linear(input_count, output_count)` | Compute `matmul(input, weight) + bias` |
| `ReLULayer::forward(input)` | Elementwise ReLU; zero gradient at nonpositive inputs |
| `Sequential::add<LayerType>(args...)` | Construct and append an owned layer |
| `Sequential::forward(input)` | Apply layers in order |
| `Sequential::parameters()` | Collect the layers' parameter handles |
| `seed(value)` | Seed the shared initialization generator (`std::uint32_t`) |
| `init::xavier_uniform(input_count, output_count)` | Return initialized weights of shape `input_count x output_count` |

Linear uses Xavier-uniform weights and zero biases. Its `parameters()` returns
weight then bias, with shapes `input_count x output_count` and
`1 x output_count`. Initialization dimensions must be positive. Both Linear
and `init::xavier_uniform` accept an optional `std::mt19937&` as their third
argument to use a caller-owned generator instead.

## Losses and optimizer

Both losses are differentiable and include the incoming loss gradient in their
backward calculation. MSE requires matching, nonempty predictions and targets.
Cross-entropy requires nonempty logits, one target per row, and target indices
less than the number of columns. Invalid loss inputs throw
`std::invalid_argument`.

`Optimizer(parameters, learning_rate)` implements plain SGD:
`parameter -= learning_rate * gradient`. `zero_grad()` clears the tracked
parameters' gradients; `step()` updates their values without clearing gradients.
Repeated handles to the same Tensor are deduplicated, so each node updates once.
The learning rate must be positive and finite, and parameter handles must be
nonnull. Invalid constructor inputs throw `std::invalid_argument`.

Batch gathering, shuffling, dataset splits, and preprocessing are handled by
caller code. See [the examples](../examples/README.md) for complete training
loops for regression and classification.
