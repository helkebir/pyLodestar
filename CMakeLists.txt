cmake_minimum_required(VERSION 3.4...3.18)
project(lodestar)

add_subdirectory(pybind11)
include_directories("src/Lodestar/src")
include_directories("src/Lodestar/dep")
include_directories("src/Lodestar/tests")
include_directories("src/Lodestar/dep/nanopb")
pybind11_add_module(lodestar src/main.cpp
                    src/Lodestar/src/analysis/ZeroOrderHold.cpp
                    src/Lodestar/src/analysis/ZeroOrderHold.hpp
                    src/Lodestar/src/analysis/BilinearTransformation.cpp
                    src/Lodestar/src/analysis/BilinearTransformation.hpp
                    src/Lodestar/src/analysis/LinearSystemInverse.cpp
                    src/Lodestar/src/analysis/LinearSystemInverse.hpp)

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(lodestar PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
