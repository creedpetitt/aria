#pragma once

#include <aria/matrix.hpp>

#include <algorithm>
#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace aria {

class Tensor;
using TensorPtr = std::shared_ptr<Tensor>;

class Tensor : public std::enable_shared_from_this<Tensor> {
public:
  Matrix data;
  Matrix grad;
  std::string op;
  std::vector<TensorPtr> prev;
  std::function<void()> backward_op;

  Tensor(aria::Matrix values, const std::string &operation = "")
      : data(std::move(values)), grad(data.rows(), data.cols(), 0.0f),
        op(operation), backward_op([] {}) {}

  Tensor(aria::Matrix values, std::vector<TensorPtr> children,
         const std::string &operation = "")
      : data(std::move(values)), grad(data.rows(), data.cols(), 0.0f),
        op(operation), prev(std::move(children)), backward_op([] {}) {}

  // backward callbacks retain this node's address. copy or move TensorPtr instead
  Tensor(const Tensor &) = delete;
  Tensor &operator=(const Tensor &) = delete;
  Tensor(Tensor &&) = delete;
  Tensor &operator=(Tensor &&) = delete;
  ~Tensor() = default;

  static TensorPtr create(aria::Matrix values,
                          const std::string &operation = "") {
    return std::make_shared<Tensor>(std::move(values), operation);
  }

  static TensorPtr create(aria::Matrix values, std::vector<TensorPtr> children,
                          const std::string &operation = "") {
    return std::make_shared<Tensor>(std::move(values), std::move(children),
                                    operation);
  }

  void backward() {
    std::vector<TensorPtr> topo;
    std::unordered_set<Tensor *> visited;

    std::function<void(TensorPtr)> build_topological = [&](TensorPtr v) {
      if (visited.contains(v.get())) {
        return;
      }
      visited.insert(v.get());
      for (const auto &input : v->prev) {
        build_topological(input);
      }
      topo.push_back(v);
    };

    build_topological(shared_from_this());
    grad = Matrix(data.rows(), data.cols(), 1.0f);

    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
      (*it)->backward_op();
    }
  }
};

inline TensorPtr matmul(const TensorPtr &a, const TensorPtr &b) {
  TensorPtr out = Tensor::create(matmul(a->data, b->data), {a, b}, "matmul");
  Tensor *out_raw = out.get();

  out->backward_op = [a, b, out_raw] {
    a->grad += matmul(out_raw->grad, transpose(b->data));
    b->grad += matmul(transpose(a->data), out_raw->grad);
  };
  return out;
}

inline TensorPtr operator+(const TensorPtr &a, const TensorPtr &b) {
  Matrix out_data = a->data + b->data;
  TensorPtr out = Tensor::create(std::move(out_data), {a, b}, "+");
  Tensor *out_raw = out.get();

  out->backward_op = [a, b, out_raw] {
    const Matrix &g = out_raw->grad;
    const auto shape = detail::broadcast_shape(a->data, b->data);
    detail::require_shape(g, shape.rows, shape.cols);
    detail::require_shape(a->grad, a->data.rows(), a->data.cols());
    detail::require_shape(b->grad, b->data.rows(), b->data.cols());
    if (shape.rows == 0 || shape.cols == 0) {
      return;
    }

    // Sum broadcast contributions into each input's original row and column.
    for (std::size_t row = 0; row < shape.rows; ++row) {
      const std::size_t a_row = a->data.rows() == 1 ? 0 : row;
      const std::size_t b_row = b->data.rows() == 1 ? 0 : row;
      for (std::size_t col = 0; col < shape.cols; ++col) {
        const std::size_t a_col = a->data.cols() == 1 ? 0 : col;
        const std::size_t b_col = b->data.cols() == 1 ? 0 : col;
        const float value = g[row, col];
        a->grad[a_row, a_col] += value;
        b->grad[b_row, b_col] += value;
      }
    }
  };
  return out;
}

// elemwise multiplication, not matmul
inline TensorPtr operator*(const TensorPtr &a, const TensorPtr &b) {
  Matrix out_data = a->data * b->data;
  TensorPtr out = Tensor::create(std::move(out_data), {a, b}, "*");
  Tensor *out_raw = out.get();

  out->backward_op = [a, b, out_raw] {
    a->grad += b->data * out_raw->grad;
    b->grad += a->data * out_raw->grad;
  };
  return out;
}

inline TensorPtr relu(const TensorPtr &a) {
  const std::size_t rows = a->data.rows();
  const std::size_t cols = a->data.cols();
  Matrix out_data(rows, cols);
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t col = 0; col < cols; ++col) {
      out_data[row, col] = std::max(0.0f, a->data[row, col]);
    }
  }

  TensorPtr out = Tensor::create(std::move(out_data), {a}, "ReLU");
  Tensor *out_raw = out.get();
  out->backward_op = [a, out_raw] {
    const std::size_t rows = a->data.rows();
    const std::size_t cols = a->data.cols();
    detail::require_shape(a->grad, rows, cols);
    detail::require_shape(out_raw->grad, rows, cols);
    for (std::size_t row = 0; row < rows; ++row) {
      for (std::size_t col = 0; col < cols; ++col) {
        if (a->data[row, col] > 0.0f) {
          a->grad[row, col] += out_raw->grad[row, col];
        }
      }
    }
  };
  return out;
}

} // namespace aria
