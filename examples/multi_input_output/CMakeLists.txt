cmake_minimum_required(VERSION 3.10)
project(multi_input_output)

add_executable(multi_input_output main.cpp)
target_link_libraries(multi_input_output cppflow)
target_compile_definitions(multi_input_output PUBLIC
  MODEL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/model"
)
