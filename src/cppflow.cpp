//
// Created by serizba on 17/9/20.
//

#include "cppflow/cppflow.h"


/******************************
 *   IMPLEMENTATION DETAILS   *
 ******************************/

namespace cppflow {
    std::string version() {
        return "TensorFlow: " + std::string(TF_Version()) + " CppFlow: 2.0.0";
    }
}

