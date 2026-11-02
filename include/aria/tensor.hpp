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

  static TensorPtr create(aria::Matrix values,
                          const std::string &operation = "") {
    return std::make_shared<Tensor>(std::move(values), operation);
  }

  static TensorPtr create(aria::Matrix values, std::vector<TensorPtr> children,
                          const std::string &operation = "") {
    return std::make_shared<Tensor>(std::move(values), std::move(children),
                                    operation);
  }

  // An all-ones seed differentiates the sum of this tensor's output entries.
  // Build a fresh forward graph for each training pass.
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
  Matrix out_data(a->data.rows(), b->data.cols());
  matmul(out_data, a->data, b->data);
  TensorPtr out = Tensor::create(std::move(out_data), {a, b}, "matmul");
  Tensor *out_raw = out.get();

  out->backward_op = [a, b, out_raw] {
    a->grad = a->grad + matmul(out_raw->grad, transpose(b->data));
    b->grad = b->grad + matmul(transpose(a->data), out_raw->grad);
  };
  return out;
}

inline TensorPtr operator+(const TensorPtr &a, const TensorPtr &b) {
  Matrix out_data = a->data + b->data;
  TensorPtr out = Tensor::create(std::move(out_data), {a, b}, "+");
  Tensor *out_raw = out.get();

  out->backward_op = [a, b, out_raw] {
    a->grad = a->grad + out_raw->grad;
    b->grad = b->grad + out_raw->grad;
  };
  return out;
}

// Elementwise multiplication, distinct from matrix multiplication.
inline TensorPtr operator*(const TensorPtr &a, const TensorPtr &b) {
  Matrix out_data = a->data * b->data;
  TensorPtr out = Tensor::create(std::move(out_data), {a, b}, "*");
  Tensor *out_raw = out.get();

  out->backward_op = [a, b, out_raw] {
    a->grad = a->grad + b->data * out_raw->grad;
    b->grad = b->grad + a->data * out_raw->grad;
  };
  return out;
}

inline TensorPtr relu(const TensorPtr &a) {
  Matrix out_data(a->data.rows(), a->data.cols());
  for (std::size_t i{}; i < out_data.rows(); ++i) {
    for (std::size_t j{}; j < out_data.cols(); ++j) {
      out_data.access_elem(i, j) = std::max(0.0f, a->data.access_elem(i, j));
    }
  }

  TensorPtr out = Tensor::create(std::move(out_data), {a}, "ReLU");
  Tensor *out_raw = out.get();
  out->backward_op = [a, out_raw] {
    for (std::size_t i{}; i < a->data.rows(); ++i) {
      for (std::size_t j{}; j < a->data.cols(); ++j) {
        if (a->data.access_elem(i, j) > 0.0f) {
          a->grad.access_elem(i, j) += out_raw->grad.access_elem(i, j);
        }
      }
    }
  };
  return out;
}

} // namespace aria
