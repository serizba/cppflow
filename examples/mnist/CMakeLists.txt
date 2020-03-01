cmake_minimum_required(VERSION 3.10)
project(example)

set(CMAKE_CXX_STANDARD 17)
find_package( OpenCV REQUIRED )
add_executable(example main.cpp ../../src/Model.cpp ../../src/Tensor.cpp)
target_include_directories(example PRIVATE ../../include)
target_link_libraries (example -ltensorflow ${OpenCV_LIBS})