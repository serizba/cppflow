cmake_minimum_required(VERSION 3.10)
project(example)

find_library(TENSORFLOW_LIB tensorflow HINT $ENV{HOME}/libtensorflow2/lib)

set(CMAKE_CXX_STANDARD 17)

add_executable(example main.cpp)
target_include_directories(example PRIVATE ../../include $ENV{HOME}/libtensorflow2/include)
target_link_libraries (example "${TENSORFLOW_LIB}")