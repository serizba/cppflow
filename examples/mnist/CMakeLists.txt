find_package(OpenCV REQUIRED )
add_executable(nmist_example main.cpp ../../src/Model.cpp ../../src/Tensor.cpp)
target_include_directories(nmist_example PRIVATE ../../include)
target_link_libraries (nmist_example ${LIBS} ${OpenCV_LIBS})
