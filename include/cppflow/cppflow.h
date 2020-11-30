//
// Created by serizba on 17/9/20.
//

#ifndef EXAMPLE_CPPFLOW_H
#define EXAMPLE_CPPFLOW_H
#include <string>
#include <tensorflow/c/c_api.h>


namespace cppflow {

    /**
     * Version of TensorFlow and CppFlow
     * @return A string containing the version of TensorFow and CppFlow
     */
    std::string version();

}
#endif //EXAMPLE_CPPFLOW_H
