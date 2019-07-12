cmake_minimum_required(VERSION 3.10)

include(CheckLibraryExists)

# Build options
set(CMAKE_CXX_STANDARD 17)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif(NOT CMAKE_BUILD_TYPE)

set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wpedantic")
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer")
endif ()

# Building
project(CppFlow VERSION 0.0.1 DESCRIPTION "C++ Flow")

set(SOURCES src/Model.cpp src/Tensor.cpp)

check_library_exists("libtensorflow.so" "" "" tensorflow_systemwide)

set(INSTALL_HEADERS "src/Tensor.h;src/Model.h")

if (tensorflow_systemwide)
    message(STATUS "Looking for tensorflow systemwide")
    set(LIBS tensorflow)
    get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
    foreach(dir ${dirs})
        message(STATUS "dir='${dir}'")
        include_directories(${CMAKE_PROJECT_NAME} PRIVATE SYSTEM dirs/tensorflow/c)
    endforeach()
else()
    message(STATUS "Looking for tensorflow in ${CMAKE_SOURCE_DIR}/libtensorflow/lib/")
    set(LIBS ${CMAKE_SOURCE_DIR}/libtensorflow/lib/libtensorflow.so)
    include_directories(${CMAKE_PROJECT_NAME} PRIVATE libtensorflow/include/tensorflow/c)
    set(INSTALL_HEADERS "${INSTALL_HEADERS};libtensorflow/include/tensorflow/c/c_api.h")
endif()

include_directories(${CMAKE_SOURCE_DIR}/src)
add_library(${PROJECT_NAME} SHARED ${SOURCES})
target_link_libraries(${PROJECT_NAME} PRIVATE ${LIBS})

# Installation via make install
include(GNUInstallDirs)
set_target_properties(${PROJECT_NAME} PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    PUBLIC_HEADER "${INSTALL_HEADERS}")
install(TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME})

add_subdirectory(examples/coco)
add_subdirectory(examples/load_model)
add_subdirectory(examples/mnist)
