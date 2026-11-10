if(NOT EMSCRIPTEN)
  find_program(ARIA_EMCMAKE_EXECUTABLE NAMES emcmake REQUIRED)
  # Keep the Emscripten compiler/cache separate from the native configuration.
  # Run its build every time so its own dependency tracking sees header edits.
  add_custom_target(aria_wasm ALL
      COMMAND "${ARIA_EMCMAKE_EXECUTABLE}" "${CMAKE_COMMAND}"
          -S "${PROJECT_SOURCE_DIR}" -B "${CMAKE_CURRENT_BINARY_DIR}/wasm"
          -G Ninja
          "-DCMAKE_BUILD_TYPE=$<CONFIG>"
          -DARIA_BUILD_WASM=ON
          "-DARIA_WASM_SIMD=${ARIA_WASM_SIMD}"
          -DARIA_BUILD_EXAMPLES=OFF
          -DARIA_BUILD_BENCHMARKS=OFF
          -DBUILD_TESTING=OFF
          -DARIA_BUILD_IO=OFF
          -DARIA_ENABLE_OPENMP=OFF
      COMMAND "${CMAKE_COMMAND}" --build "${CMAKE_CURRENT_BINARY_DIR}/wasm"
          --target aria_wasm
      COMMENT "Updating JavaScript/WebAssembly module"
      USES_TERMINAL
      VERBATIM)
  return()
endif()

add_executable(aria_wasm ${PROJECT_SOURCE_DIR}/wasm/wasm.cpp)
target_link_libraries(aria_wasm PRIVATE aria::aria)
target_compile_options(aria_wasm PRIVATE -fwasm-exceptions)
if(ARIA_WASM_SIMD)
  target_compile_options(aria_wasm PRIVATE -msimd128)
  target_link_options(aria_wasm PRIVATE -msimd128)
endif()
target_link_options(aria_wasm PRIVATE
    -lembind
    --no-entry
    -fwasm-exceptions
    -sMODULARIZE=1
    -sEXPORT_ES6=1
    -sEXPORT_NAME=createAriaModule
    -sALLOW_MEMORY_GROWTH=1
    "SHELL:--emit-tsd \"${CMAKE_CURRENT_BINARY_DIR}/aria.d.mts\"")
set_target_properties(aria_wasm PROPERTIES OUTPUT_NAME aria SUFFIX ".mjs")
