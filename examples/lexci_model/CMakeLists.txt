cmake_minimum_required(VERSION 3.10)
project(lexci_model)

find_library(TENSORFLOW_LIB tensorflow HINT $ENV{HOME}/libtensorflow2/lib)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer -fsanitize=address")
set(CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} -lasan")

add_executable(lexci_model main.cpp)
target_include_directories(lexci_model PRIVATE ../../include $ENV{HOME}/libtensorflow2/include)
target_link_libraries (lexci_model "${TENSORFLOW_LIB}")
target_compile_definitions(lexci_model PUBLIC
  IMAGE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/model/TNG50-1_halpha.jpeg"
  MODEL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/model"
)
