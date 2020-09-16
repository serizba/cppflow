cmake_minimum_required(VERSION 3.10)
project(example)

find_library(TENSORFLOW_LIB tensorflow HINT ../../libtensorflow2/lib)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-g -O3 -Wall -Wextra")


add_executable(example main.cpp)
target_include_directories(example PRIVATE ../../include ../../libtensorflow2/include)
target_link_libraries (example "${TENSORFLOW_LIB}")