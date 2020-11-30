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
    TF_DataType deduce_tf_type() {
        if (std::is_same<T, float>::value)
            return TF_FLOAT;
        if (std::is_same<T, double>::value)
            return TF_DOUBLE;
        if (std::is_same<T, int32_t >::value)
            return TF_INT32;
        if (std::is_same<T, uint8_t>::value)
            return TF_UINT8;
        if (std::is_same<T, int16_t>::value)
            return TF_INT16;
        if (std::is_same<T, int8_t>::value)
            return TF_INT8;
        if (std::is_same<T, int64_t>::value)
            return TF_INT64;
        if (std::is_same<T, unsigned char>::value)
            return TF_BOOL;
        if (std::is_same<T, uint16_t>::value)
            return TF_UINT16;
        if (std::is_same<T, uint32_t>::value)
            return TF_UINT32;
        if (std::is_same<T, uint64_t>::value)
            return TF_UINT64;

        throw std::runtime_error{ "Could not deduce type!" };
    }
	
    /**
     * @return  The stream os after inserting the string representation of dt
     *
     */
    std::ostream& operator<<(std::ostream& os, datatype dt);
    }

#endif //CPPFLOW2_DATATYPE_H
