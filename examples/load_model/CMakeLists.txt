set(CMAKE_CXX_STANDARD 17)
add_executable(load_model_example main.cpp ../../src/Model.cpp ../../src/Tensor.cpp)
target_include_directories(load_model_example PRIVATE ../../include)
target_link_libraries (load_model_example ${LIBS})
