cmake_minimum_required(VERSION 3.10)
project(eager_op_multithread)

find_package(Threads REQUIRED)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer -fsanitize=thread")
#set(CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} -fno-omit-frame-pointer -fsanitize=thread")

add_executable(eager_op_multithread main.cpp)
target_link_libraries(eager_op_multithread Threads::Threads cppflow)
