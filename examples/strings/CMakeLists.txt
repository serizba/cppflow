cmake_minimum_required(VERSION 3.10)
project(strings)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer -fsanitize=address")
set(CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} -lasan")

add_executable(strings main.cpp)
target_link_libraries(strings cppflow)
