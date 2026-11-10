#include <aria.hpp>
#include <emscripten/bind.h>
#include <emscripten/val.h>

using namespace emscripten;
using aria::Matrix;
using aria::Tensor;
using aria::TensorPtr;

EMSCRIPTEN_DECLARE_VAL_TYPE(FloatArray);
EMSCRIPTEN_DECLARE_VAL_TYPE(ClassArray);
EMSCRIPTEN_DECLARE_VAL_TYPE(TensorArray);

namespace {

void require_array(const val &value, const char *type) {
  if (!value.instanceof(val::global(type))) {
    throw std::invalid_argument(std::string("Expected ") + type);
  }
}

// views exist only during these calls, public arrays own independent JS storage
void copy_from(Matrix &matrix, const FloatArray &values) {
  require_array(values, "Float32Array");
  const std::size_t count = matrix.rows() * matrix.cols();
  if (values["length"].as<std::size_t>() != count) {
    throw std::invalid_argument("Array length must match matrix dimensions");
  }
  val(typed_memory_view(count, matrix.data())).call<void>("set", values);
}

Matrix from_array(std::size_t rows, std::size_t cols, const FloatArray &values) {
  require_array(values, "Float32Array");
  if (static_cast<double>(rows) * cols != values["length"].as<double>()) {
    throw std::invalid_argument("Array length must match matrix dimensions");
  }
  Matrix matrix(rows, cols);
  copy_from(matrix, values);
  return matrix;
}

FloatArray to_array(const Matrix &matrix) {
  return FloatArray(val::global("Float32Array").new_(
      val(typed_memory_view(matrix.rows() * matrix.cols(), matrix.data()))));
}

const TensorPtr &require_tensor(const TensorPtr &tensor) {
  if (!tensor) throw std::invalid_argument("Tensor must not be null");
  return tensor;
}

TensorArray parameters(const aria::Layer &layer) {
  val result = val::array();
  for (const auto &parameter : layer.parameters()) {
    result.call<void>("push", parameter);
  }
  return TensorArray(result);
}

aria::Optimizer *make_optimizer(const TensorArray &values, float learning_rate) {
  if (!val::global("Array").call<bool>("isArray", values)) {
    throw std::invalid_argument("Optimizer requires an array of Tensor handles");
  }
  return new aria::Optimizer(vecFromJSArray<TensorPtr>(values), learning_rate);
}

} // namespace

EMSCRIPTEN_BINDINGS(aria_api) {
  register_type<FloatArray>("Float32Array");
  register_type<ClassArray>("Uint32Array");
  register_type<TensorArray>("Tensor[]");

  class_<Matrix>("Matrix")
      .constructor<std::size_t, std::size_t>()
      .class_function("fromArray", &from_array)
      .function("copyFrom", &copy_from)
      .function("toArray", &to_array)
      .function("rows", &Matrix::rows)
      .function("cols", &Matrix::cols)
      .function("fill", &Matrix::fill)
      .function("get", optional_override(
          [](const Matrix &m, std::size_t row, std::size_t col) {
            return m[row, col];
          }))
      .function("set", optional_override(
          [](Matrix &m, std::size_t row, std::size_t col, float value) {
            m[row, col] = value;
          }));

  function("add", select_overload<Matrix(const Matrix &, const Matrix &)>(&aria::operator+));
  function("subtract", select_overload<Matrix(const Matrix &, const Matrix &)>(&aria::operator-));
  function("multiply", select_overload<Matrix(const Matrix &, const Matrix &)>(&aria::operator*));
  function("scale", select_overload<Matrix(const Matrix &, float)>(&aria::operator*));
  function("matmul", select_overload<Matrix(const Matrix &, const Matrix &)>(&aria::matmul));
  function("transpose", select_overload<Matrix(const Matrix &)>(&aria::transpose));
  function("softmax", select_overload<Matrix(const Matrix &)>(&aria::softmax));

  function("addInto", &aria::add);
  function("subtractInto", &aria::subtract);
  function("multiplyInto", &aria::multiply);
  function("scaleInto", &aria::scale);
  function("matmulInto", select_overload<void(Matrix &, const Matrix &, const Matrix &)>(&aria::matmul));
  function("transposeInto", select_overload<void(Matrix &, const Matrix &)>(&aria::transpose));
  function("softmaxInto", select_overload<void(Matrix &, const Matrix &)>(&aria::softmax));

  // all Tensor results retain shared ownership; no Tensor nodes are copied or moved
  class_<Tensor>("Tensor")
      .smart_ptr<TensorPtr>("TensorPtr")
      .class_function("create", +[](const Matrix &values) {
        return Tensor::create(values);
      }, nonnull<ret_val>())
      .class_function("fromArray", +[](std::size_t rows, std::size_t cols, const FloatArray &values) {
        return Tensor::create(from_array(rows, cols, values));
      }, nonnull<ret_val>())
      .function("rows", optional_override([](const Tensor &t) { return t.data.rows(); }))
      .function("cols", optional_override([](const Tensor &t) { return t.data.cols(); }))
      .function("toArray", optional_override([](const Tensor &t) { return to_array(t.data); }))
      .function("gradToArray", optional_override([](const Tensor &t) { return to_array(t.grad); }))
      .function("copyFrom", optional_override([](Tensor &t, const FloatArray &values) {
        copy_from(t.data, values);
      }))
      .function("item", optional_override([](const Tensor &t) {
        if (t.data.rows() != 1 || t.data.cols() != 1) {
          throw std::invalid_argument("item() requires a 1 x 1 Tensor");
        }
        return t.data[0, 0];
      }))
      .function("zeroGrad", optional_override([](Tensor &t) { t.grad.fill(0); }))
      .function("backward", &Tensor::backward);

  function("tensorAdd", +[](const TensorPtr &a, const TensorPtr &b) {
    return require_tensor(a) + require_tensor(b);
  }, nonnull<ret_val>());
  function("tensorMultiply", +[](const TensorPtr &a, const TensorPtr &b) {
    return require_tensor(a) * require_tensor(b);
  }, nonnull<ret_val>());
  function("tensorMatmul", +[](const TensorPtr &a, const TensorPtr &b) {
    return aria::matmul(require_tensor(a), require_tensor(b));
  }, nonnull<ret_val>());
  function("relu", +[](const TensorPtr &input) {
    return aria::relu(require_tensor(input));
  }, nonnull<ret_val>());

  class_<aria::Layer>("Layer")
      .function("forward", optional_override([](const aria::Layer &layer, const TensorPtr &input) {
        return layer.forward(require_tensor(input));
      }), nonnull<ret_val>())
      .function("parameters", &parameters);
  class_<aria::Linear, base<aria::Layer>>("Linear")
      .constructor<std::size_t, std::size_t>();
  class_<aria::ReLULayer, base<aria::Layer>>("ReLULayer").constructor<>();
  class_<aria::Sequential, base<aria::Layer>>("Sequential")
      .constructor<>()
      .function("addLinear", optional_override([](aria::Sequential &model, std::size_t in, std::size_t out) {
        model.add<aria::Linear>(in, out);
      }))
      .function("addReLU", optional_override([](aria::Sequential &model) {
        model.add<aria::ReLULayer>();
      }));

  class_<aria::Optimizer>("Optimizer")
      .constructor(&make_optimizer, allow_raw_pointers())
      .function("zeroGrad", &aria::Optimizer::zero_grad)
      .function("step", &aria::Optimizer::step);
  function("meanSquaredError", &aria::mean_squared_error, nonnull<ret_val>());
  function("crossEntropy", +[](const TensorPtr &logits, const ClassArray &classes) {
    require_array(classes, "Uint32Array");
    return aria::combined_cross_entropy(logits, convertJSArrayToNumberVector<std::size_t>(classes));
  }, nonnull<ret_val>());
  function("seed", &aria::seed);
  function("xavierUniform", +[](std::size_t inputs, std::size_t outputs) {
    return aria::init::xavier_uniform(inputs, outputs);
  });
}
