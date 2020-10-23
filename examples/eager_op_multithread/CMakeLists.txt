cmake_minimum_required(VERSION 3.10)
project(example)

find_library(TENSORFLOW_LIB tensorflow HINT $ENV{HOME}/libtensorflow2/lib)
find_package(Threads REQUIRED)

set(CMAKE_CXX_STANDARD 17)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer -fsanitize=thread")
#set(CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} -fno-omit-frame-pointer -fsanitize=thread")

add_executable(example main.cpp)
target_include_directories(example PRIVATE ../../include $ENV{HOME}/libtensorflow2/include)
target_link_libraries(example "${TENSORFLOW_LIB}" Threads::Threads)
