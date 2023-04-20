cmake_minimum_required(VERSION 3.10)

project(DNNInferencer_blas)

add_library(${PROJECT_NAME} SHARED DNNInferencer_blas.cpp Layer.cpp)

# Link the TensorFlow library
target_link_libraries(
    ${PROJECT_NAME}
        fjlapackexsve)