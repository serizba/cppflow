cmake_minimum_required(VERSION 3.10)
project(load_model)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer -fsanitize=address")
set(CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} -lasan")

add_executable(load_model main.cpp)
target_link_libraries(load_model cppflow)
target_compile_definitions(load_model PUBLIC
  MODEL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/model"
)
