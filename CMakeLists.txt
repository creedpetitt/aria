cmake_minimum_required(VERSION 3.21)

include(cmake/AriaOptions.cmake)
project(aria LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(aria INTERFACE)
add_library(aria::aria ALIAS aria)
target_include_directories(aria INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(aria INTERFACE cxx_std_23)

option(ARIA_BUILD_EXAMPLES "Build MNIST and enabled tabular examples" OFF)
option(ARIA_BUILD_BENCHMARKS "Build matrix benchmarks" OFF)
option(ARIA_BUILD_WASM "Build JavaScript/WebAssembly bindings" OFF)
option(ARIA_WASM_SIMD "Enable 128-bit SIMD in WebAssembly bindings" ON)
option(ARIA_ENABLE_OPENMP "Parallelize native matrix multiplication with OpenMP" OFF)
option(BUILD_TESTING "Build tests" OFF)

if(ARIA_ENABLE_OPENMP)
  if(EMSCRIPTEN)
    message(FATAL_ERROR "ARIA_ENABLE_OPENMP is native-only; disable it for Emscripten")
  endif()
  find_package(OpenMP REQUIRED COMPONENTS CXX)
  target_link_libraries(aria INTERFACE OpenMP::OpenMP_CXX)
  target_compile_definitions(aria INTERFACE ARIA_USE_OPENMP=1)
endif()

if(ARIA_BUILD_WASM)
  include(cmake/AriaWasm.cmake)
endif()

if(ARIA_BUILD_EXAMPLES)
  add_executable(math main.cpp)
  target_link_libraries(math PRIVATE aria::aria)
endif()

include(CTest)
if(ARIA_BUILD_IO)
  include(cmake/AriaIO.cmake)
  if(ARIA_BUILD_EXAMPLES AND ARIA_IO_CSV)
    foreach(dataset IN ITEMS housing titanic)
      add_executable(${dataset}_example examples/${dataset}.cpp)
      target_link_libraries(${dataset}_example PRIVATE aria::io)
      if(BUILD_TESTING AND
         EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/data/tabular/${dataset}/train.csv" AND
         EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/data/tabular/${dataset}/test.csv")
        add_test(NAME ${dataset}_training COMMAND ${dataset}_example
                 "${CMAKE_CURRENT_SOURCE_DIR}/data/tabular/${dataset}")
        set_tests_properties(${dataset}_training PROPERTIES TIMEOUT 120)
      endif()
    endforeach()
  endif()
endif()

if(BUILD_TESTING)
  add_executable(matrix_tests tests/matrix_tests.cpp)
  target_link_libraries(matrix_tests PRIVATE aria::aria)
  add_test(NAME matrix_tests COMMAND matrix_tests)

  add_executable(tensor_tests tests/tensor_tests.cpp)
  target_link_libraries(tensor_tests PRIVATE aria::aria)
  add_test(NAME tensor_tests COMMAND tensor_tests)

  add_executable(mnist_tests tests/mnist_tests.cpp)
  target_link_libraries(mnist_tests PRIVATE aria::aria)
  target_include_directories(mnist_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  add_test(NAME mnist_loader_tests COMMAND mnist_tests)

  if(ARIA_BUILD_EXAMPLES AND
     EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/data/mnist/train-images-idx3-ubyte" AND
     EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/data/mnist/train-labels-idx1-ubyte" AND
     EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/data/mnist/t10k-images-idx3-ubyte" AND
     EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/data/mnist/t10k-labels-idx1-ubyte")
    add_test(NAME mnist_training COMMAND math "${CMAKE_CURRENT_SOURCE_DIR}/data/mnist")
    set_tests_properties(mnist_training PROPERTIES TIMEOUT 120)
  endif()
endif()

if(ARIA_BUILD_BENCHMARKS)
  add_executable(matrix_bench benchmarks/matrix_bench.cpp)
  target_link_libraries(matrix_bench PRIVATE aria::aria)
endif()
