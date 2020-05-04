cmake_minimum_required(VERSION 3.10)
project(example)

find_library(TENSORFLOW_LIB tensorflow HINT $ENV{HOME}/libtensorflow/lib)

set(CMAKE_CXX_STANDARD 17)
find_package( OpenCV REQUIRED )
add_executable(example main.cpp ../../src/Model.cpp ../../src/Tensor.cpp)
target_include_directories(example PRIVATE ../../include $ENV{HOME}/libtensorflow/include)
target_link_libraries (example "${TENSORFLOW_LIB}" ${OpenCV_LIBS})
