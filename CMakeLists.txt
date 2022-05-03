cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(cppflow LANGUAGES CXX)

include(GNUInstallDirs)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME}/cmake)

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_SOURCE_DIR}/cmake/modules")


# Library target
find_package(tensorflow REQUIRED)
add_library(cppflow INTERFACE)
target_include_directories(cppflow
  INTERFACE
    ${tensorflow_INCLUDE_DIRS}
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(cppflow INTERFACE cxx_std_17)
target_link_libraries(cppflow INTERFACE
  ${tensorflow_LIBRARIES}
)
install(
  TARGETS cppflow
  EXPORT install_targets
)


# Build examples
option(BUILD_EXAMPLES "Build examples" ON)
if(BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()


# Install headers
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})


# Install targets file
install(EXPORT install_targets
  FILE
    ${PROJECT_NAME}Targets.cmake
  NAMESPACE
    ${PROJECT_NAME}::
  DESTINATION
    ${INSTALL_CONFIGDIR}
)


# Install cppflowConfig.cmake
include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
  INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
)
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
  DESTINATION ${INSTALL_CONFIGDIR}
)


# Install find modules
install(DIRECTORY cmake/modules/ DESTINATION ${INSTALL_CONFIGDIR}/modules)
