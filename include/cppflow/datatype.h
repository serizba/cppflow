//
// Created by serizba on 12/7/20.
//

#ifndef CPPFLOW2_DATATYPE_H
#define CPPFLOW2_DATATYPE_H

#include <string>
#include <tensorflow/c/tf_datatype.h>
#include <ostream>

namespace cppflow {

    using datatype = TF_DataType;

    /**
     * @return A string representing dt
     *
     */
    std::string to_string(datatype dt);

    /**
     *
     * @tparam T
     * @return The TensorFlow type of T
     */
    template<typename T>
    TF_DataType deduce_tf_type();

    /**
     * @return  The stream os after inserting the string representation of dt
     *
     */
    std::ostream& operator<<(std::ostream& os, datatype dt);
    }

#endif //CPPFLOW2_DATATYPE_H
