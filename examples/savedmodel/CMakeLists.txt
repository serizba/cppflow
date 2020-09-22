cmake_minimum_required(VERSION 3.10)
project(example)

if(NOT TF_API_DIR)
set(TF_API_DIR $ENV{HOME}/libtensorflow)
endif(NOT TF_API_DIR)

find_library(TENSORFLOW_LIB tensorflow HINT ${TF_API_DIR}/lib)

set(CMAKE_CXX_STANDARD 17)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR})
add_executable(example main.cpp ../../src/Model.cpp ../../src/Tensor.cpp ../../src/SavedModel.cpp)
target_include_directories(example PRIVATE ../../include ${TF_API_DIR}/include)
target_link_libraries (example "${TENSORFLOW_LIB}")
