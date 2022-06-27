cmake_minimum_required(VERSION 3.10)
project(multi_input_output)

find_library(TENSORFLOW_LIB tensorflow)

set(CMAKE_CXX_STANDARD 17)

add_executable(multi_input_output main.cpp)
target_include_directories(multi_input_output PRIVATE ../../include $ENV{HOME}/libtensorflow2/include)
target_link_libraries (multi_input_output "${TENSORFLOW_LIB}")
target_compile_definitions(multi_input_output PUBLIC
  MODEL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/model"
)
