cmake_minimum_required(VERSION 3.10)
project(efficientnet)

find_library(TENSORFLOW_LIB tensorflow HINT $ENV{HOME}/libtensorflow2/lib)

set(CMAKE_CXX_STANDARD 17)

add_executable(efficientnet main.cpp)
target_include_directories(efficientnet PRIVATE ../../include $ENV{HOME}/libtensorflow2/include)
target_link_libraries (efficientnet "${TENSORFLOW_LIB}")
target_compile_definitions(efficientnet PUBLIC
  CAT_PATH="${CMAKE_CURRENT_SOURCE_DIR}/my_cat.jpg"
  MODEL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/model"
)
