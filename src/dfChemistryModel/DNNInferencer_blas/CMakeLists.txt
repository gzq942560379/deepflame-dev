cmake_minimum_required(VERSION 3.10)

project(DNNInferencer_blas)

find_package(OpenMP REQUIRED)

include_directories(
    /home/jx278/yaml-cpp/include
)

set(CMAKE_CXX_COMPILER mpiCC)

set(CMAKE_CXX_FLAGS "-Ofast -g -fopenmp -std=c++11")

add_library(${PROJECT_NAME} SHARED DNNInferencer_blas.cpp Layer.cpp kernel_double.cpp kernel_float.cpp DNNThermo_blas.cpp)

target_link_libraries(${PROJECT_NAME}   
    /home/jx278/yaml-cpp/build/libyaml-cpp.a
)
