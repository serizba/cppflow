cmake_minimum_required(VERSION 3.10)
project(tensor)

add_executable(tensor main.cpp odr.cpp)
target_link_libraries(tensor cppflow)
