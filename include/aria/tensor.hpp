#pragma once

#include <aria/autograd_ops.hpp>
#include <aria/matrix.hpp>

#include <algorithm>
#include <cstddef>
#include <functional>
#include <memory>
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
  AutogradOp op;
  std::vector<TensorPtr> prev;
  std::function<void()> backward_op;

  Tensor(aria::Matrix values, AutogradOp operation = AutogradOp::None)
      : data(std::move(values)), grad(data.rows(), data.cols(), 0.0f),
        op(operation), backward_op([] {}) {}

  Tensor(aria::Matrix values, std::vector<TensorPtr> children,
         AutogradOp operation = AutogradOp::None)
      : data(std::move(values)), grad(data.rows(), data.cols(), 0.0f),
        op(operation), prev(std::move(children)), backward_op([] {}) {}

  // backward callbacks retain this node's address. copy or move TensorPtr instead
  Tensor(const Tensor &) = delete;
  Tensor &operator=(const Tensor &) = delete;
  Tensor(Tensor &&) = delete;
  Tensor &operator=(Tensor &&) = delete;
  ~Tensor() = default;

  static TensorPtr create(aria::Matrix values,
                          const AutogradOp &operation = AutogradOp::None) {
    return std::make_shared<Tensor>(std::move(values), operation);
  }

  static TensorPtr create(aria::Matrix values, std::vector<TensorPtr> children,
                          const AutogradOp &operation) {
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
  Matrix out_data = matmul(a->data, b->data);
  TensorPtr out = Tensor::create(std::move(out_data), {a, b}, AutogradOp::Matmul);
  Tensor *out_raw = out.get();

  out->backward_op = [a, b, out_raw] {
    autograd::matmul_backward(a->grad, b->grad, a->data, b->data,
                              out_raw->grad);
  };
  return out;
}

inline TensorPtr operator+(const TensorPtr &a, const TensorPtr &b) {
  Matrix out_data = a->data + b->data;
  TensorPtr out = Tensor::create(std::move(out_data), {a, b}, AutogradOp::Add);
  Tensor *out_raw = out.get();

  out->backward_op = [a, b, out_raw] {
    autograd::add_backward(a->grad, b->grad, a->data, b->data, out_raw->grad);
  };
  return out;
}

// elemwise multiplication, not matmul
inline TensorPtr operator*(const TensorPtr &a, const TensorPtr &b) {
  Matrix out_data = a->data * b->data;
  TensorPtr out = Tensor::create(std::move(out_data), {a, b}, AutogradOp::Multiply);
  Tensor *out_raw = out.get();

  out->backward_op = [a, b, out_raw] {
    autograd::multiply_backward(a->grad, b->grad, a->data, b->data,
                                out_raw->grad);
  };
  return out;
}

// scalar multiplication
inline TensorPtr operator*(const TensorPtr &a, const float scalar) {
  Matrix out_data = a->data * scalar;
  TensorPtr out = Tensor::create(std::move(out_data), {a}, AutogradOp::Scale);
  Tensor *out_raw = out.get();

  out->backward_op = [a, scalar, out_raw] {
    autograd::scale_backward(a->grad, out_raw->grad, scalar);
  };
  return out;
}

inline TensorPtr transpose(const TensorPtr &a) {
  Matrix out_data = transpose(a->data);
  TensorPtr out = Tensor::create(std::move(out_data), {a}, AutogradOp::Transpose);
  Tensor *out_raw = out.get();

  out->backward_op = [a, out_raw] {
    autograd::transpose_backward(a->grad, out_raw->grad);
  };

  return out;
}

inline TensorPtr softmax(const TensorPtr &a) {
  Matrix out_data = softmax(a->data);
  TensorPtr out = Tensor::create(std::move(out_data), {a}, AutogradOp::Softmax);
  Tensor *out_raw = out.get();

  out->backward_op = [a, out_raw] {
    autograd::softmax_backward(a->grad, out_raw->data, out_raw->grad);
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

  TensorPtr out = Tensor::create(std::move(out_data), {a}, AutogradOp::ReLU);
  Tensor *out_raw = out.get();
  out->backward_op = [a, out_raw] {
    autograd::relu_backward(a->grad, a->data, out_raw->grad);
  };
  return out;
}

} // namespace aria
