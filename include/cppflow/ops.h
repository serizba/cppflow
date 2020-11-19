//
// Created by serizba on 31/7/20.
//

#ifndef CPPFLOW2_OPS_H
#define CPPFLOW2_OPS_H


#include "tensor.h"
#include "raw_ops.h"

namespace cppflow {

    /**
     * @name Operators
     */
    //@{

    /**
     * @returns x + y elementwise
     */
    tensor operator+(const tensor& x, const tensor& y);

    /**
     * @returns x - y elementwise
     */
    tensor operator-(const tensor& x, const tensor& y);

    /**
     * @returns x * y elementwise
     */
    tensor operator*(const tensor& x, const tensor& y);

    /**
     * @return x / y elementwise
     */
    tensor operator/(const tensor& x, const tensor& y);

    std::ostream& operator<<(std::ostream& os, const cppflow::tensor& t);

    //@}

    /**
     * @return A string representing t in the form:
     * (tensor: shape=?, data=
     * ?)
     */
    std::string to_string(const tensor& t);
}

#endif //CPPFLOW2_OPS_H
