from conan import ConanFile
from conan.tools.cmake import CMake, CMakeToolchain, CMakeDeps
from conan.tools.files import symlinks
import os

'''
Create the package with conan create . -pr:h=default -pr:b=default
'''
class CppflowConan(ConanFile):
    name = "cppflow"
    version = "2.0.0"
    url = "https://github.com/serizba/cppflow"
    license = "MIT"
    description = "Run TensorFlow models in c++ without Bazel, without TensorFlow installation and without compiling Tensorflow. Perform tensor manipulation, use eager execution and run saved models directly from C++."
    settings = "os", "compiler", "build_type", "arch"
    generators = "CMakeDeps"
    exports_sources = "CMakeLists.txt", "cmake/*", "include/*", "examples/*"

    def generate(self):

        tc = CMakeToolchain(self)
        tc.generate()
        tc.variables['CMAKE_VERBOSE_MAKEFILE:BOOL'] = "ON"
        deps = CMakeDeps(self)
        deps.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.install()

        
