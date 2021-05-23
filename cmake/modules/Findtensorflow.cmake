find_path(tensorflow_INCLUDE_DIRS
  NAMES tensorflow/c/c_api.h
)
mark_as_advanced(tensorflow_INCLUDE_DIRS)

find_library(tensorflow_LIBRARIES
  NAMES tensorflow
)
mark_as_advanced(tensorflow_LIBRARIES)


if(NOT tensorflow_INCLUDE_DIRS)
  message(STATUS "Could NOT find tensorflow/c/c_api.h")
endif()
if(NOT tensorflow_LIBRARIES)
  message(STATUS "Could NOT find tensorflow library")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(tensorflow DEFAULT_MSG tensorflow_INCLUDE_DIRS tensorflow_LIBRARIES)
