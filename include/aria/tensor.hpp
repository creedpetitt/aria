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
    const Matrix &g = out_raw->grad;

    for (size_t row{}; row < g.rows(); ++row) {
      for (size_t col{}; col < g.cols(); ++col) {
        auto a_row = row;
        auto a_col = col;
        auto b_row = row;
        auto b_col = col;

        if (a->data.rows() == 1)
          a_row = 0;
        if (a->data.cols() == 1)
          a_col = 0;
        if (b->data.rows() == 1)
          b_row = 0;
        if (b->data.cols() == 1)
          b_col = 0;

        a->grad[a_row, a_col] = a->grad[a_row, a_col] + g[row, col];
        b->grad[b_row, b_col] = b->grad[b_row, b_col] + g[row, col];
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
