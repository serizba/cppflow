cmake_minimum_required(VERSION 3.10)
project(example)

set(CMAKE_CXX_STANDARD 17)
find_library(TENSORFLOW_LIB tensorflow HINT $ENV{HOME}/libtensorflow2/lib)
find_package( OpenCV REQUIRED )

add_executable(example main.cpp)
include_directories( ${OpenCV_INCLUDE_DIRS} )
target_include_directories(example PRIVATE ../../include $ENV{HOME}/libtensorflow2/include)
target_link_libraries (example -ltensorflow)
target_link_libraries(example ${OpenCV_LIBS} )
