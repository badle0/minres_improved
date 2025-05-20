cmake_minimum_required(VERSION 3.10)
project(MinresExample LANGUAGES CXX)

# 1) Specify the C++ standard:
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 2) Set ONNXRuntime include/library paths if needed:
set(ONNXRUNTIME_INCLUDE_DIR "/data/hsheng/virtualenvs/onnxruntime/onnxruntime-linux-x64-gpu-1.20.0/include")
set(ONNXRUNTIME_LIBRARY "/data/hsheng/virtualenvs/onnxruntime/onnxruntime-linux-x64-gpu-1.20.0/lib/libonnxruntime.so")

# 3) Gather your source files for the minres_test executable
file(GLOB MINRES_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/helpers/helper_functions.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ONNXModelPredictor.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lstsquares_example/LstsquaresVector.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lstsquares_example/LstsquaresOperator.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lstsquares_example/SOperator.cpp"
)

# 4) Create the minres_test executable
add_executable(minres_test ${MINRES_SOURCES})

# 5) Include directories so the compiler finds your headers
target_include_directories(minres_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}                     # Project root
    ${CMAKE_CURRENT_SOURCE_DIR}/lstsquares_example    # Your vector/operator classes
    ${CMAKE_CURRENT_SOURCE_DIR}/helpers
    ${ONNXRUNTIME_INCLUDE_DIR}                        # ONNXRuntime headers
    "${CMAKE_CURRENT_SOURCE_DIR}/matrix-factor"       # SYM-ILDL: contains source/solver.h
)

# 6) Link the ONNXRuntime library (if your code references it)
target_link_libraries(minres_test PRIVATE ${ONNXRUNTIME_LIBRARY})

# 7) Enable debug flags for minres_test, including AddressSanitizer:
target_compile_options(minres_test PRIVATE -fsanitize=address -g -O0 -Wall -Wextra -pedantic)
target_link_options(minres_test PRIVATE -fsanitize=address)
