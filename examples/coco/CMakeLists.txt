add_executable(coco_example main.cpp ../../src/Model.cpp ../../src/Tensor.cpp)
find_package(OpenCV REQUIRED )
target_include_directories(coco_example PRIVATE ../../include)
target_link_libraries (coco_example ${LIBS} ${OpenCV_LIBS})
