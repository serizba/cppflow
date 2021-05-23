cmake_minimum_required(VERSION 3.10)
project(efficientnet)

add_executable(efficientnet main.cpp)
target_link_libraries(efficientnet cppflow)
target_compile_definitions(efficientnet PUBLIC
  CAT_PATH="${CMAKE_CURRENT_SOURCE_DIR}/my_cat.jpg"
  MODEL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/model"
)
