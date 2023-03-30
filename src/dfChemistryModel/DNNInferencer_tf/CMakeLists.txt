# cmake_minimum_required(VERSION 3.0)

# set(CMAKE_PREFIX_PATH $ENV{LIBTORCH_ROOT})
# project(DNNInferencer)
# find_package(Torch REQUIRED)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
# add_library(DNNInferencer SHARED DNNInferencer.cpp)
# target_compile_features(DNNInferencer PUBLIC cxx_range_for)
# target_link_libraries(DNNInferencer ${TORCH_LIBRARIES})
# set_property(TARGET DNNInferencer PROPERTY CXX_STANDARD 14)

# set(CMAKE_INSTALL_PREFIX $ENV{DF_ROOT})
# install (TARGETS ${PROJECT_NAME}
#     LIBRARY DESTINATION lib
#     )


cmake_minimum_required(VERSION 3.10)

project(DNNInferencertf)

# Set the path to the TensorFlow C++ library
set(TENSORFLOW_LIB_PATH "/home/runze/tensorflow/tensorflow/bazel-bin/tensorflow")
set(TENSORFLOW_INCLUDE_PATH "/home/runze/tensorflow/tensorflow")

# Find the TensorFlow libraries
# find_library(TENSORFLOW_LIB tensorflow HINTS ${TENSORFLOW_LIB_PATH})

# Add the include directories
include_directories(
    ${TENSORFLOW_INCLUDE_PATH})

# Create the executable
add_library(DNNInferencertf SHARED DNNInferencertf.cpp)

# Link the TensorFlow library
target_link_libraries(
    ${PROJECT_NAME}
        ${TENSORFLOW_LIB_PATH}/libtensorflow_cc.so)