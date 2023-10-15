cmake_minimum_required(VERSION 3.10)

project(DNNInferencer_blas)

add_library(${PROJECT_NAME} STATIC DNNInferencer_blas.cpp Layer.cpp)

option(BLAS_LIBRARY "blas libraries" "")

if(NOT BLAS_LIBRARY)
    message(FATAL_ERROR "BLAS_LIBRARY is not set !!!")
endif()

# Link the TensorFlow library
target_link_libraries(${PROJECT_NAME} ${BLAS_LIBRARY})