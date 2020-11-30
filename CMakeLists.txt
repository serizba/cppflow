# CMakeList.txt : projet CMake pour tensorflow_lite_test, incluez la source et définissez
# la logique spécifique au projet ici.
#


cmake_minimum_required (VERSION 3.16)

set(CMAKE_CXX_STANDARD 11)

project (cppflow)

find_library(TENSORFLOW_LIB tensorflow HINT $ENV{HOME}/libtensorflow2/lib)

target_include_directories(cppflow PRIVATE ../../include $ENV{HOME}/libtensorflow2/include)

# Ajoutez une source à l'exécutable de ce projet.
add_library(cppflow STATIC "src/context.cpp" "src/cppflow.cpp" "src/datatype.cpp" "src/model.cpp" "src/tensor.cpp" "src/raw_ops.cpp" "src/ops.cpp")

#set_property(TARGET cppflow2 PROPERTY
#        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")


target_link_libraries(cppflow
        tensorflow_lite
        )


# TODO: Ajoutez des tests et installez des cibles si nécessaire.
