// MIT License
//
// Copyright (c) 2020 Sergio Izquierdo
// Copyright (c) 2020 Jiannan Liu
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

/*!
 *  @file       cppflow.h
 *  @author     Jiannan Liu
 *  @author     Sergio Izquierdo
 *  @date       @showdate "%B %d, %Y" 2020-09-17
 */

#ifndef INCLUDE_CPPFLOW_CPPFLOW_H_
#define INCLUDE_CPPFLOW_CPPFLOW_H_

// C headers
#include <tensorflow/c/c_api.h>

// C++ headers
#include <string>

// CppFlow headers
#include "cppflow/datatype.h"
#include "cppflow/model.h"
#include "cppflow/ops.h"
#include "cppflow/raw_ops.h"
#include "cppflow/tensor.h"

namespace cppflow {

/**
 * Version of TensorFlow and CppFlow
 * @return A string containing the version of TensorFow and CppFlow
 */
std::string version();

}  // namespace cppflow

/******************************
 *   IMPLEMENTATION DETAILS   *
 ******************************/

namespace cppflow {

inline std::string version() {
  return "TensorFlow: " + std::string(TF_Version()) + " CppFlow: 2.0.0";
}

}  // namespace cppflow

#endif  // INCLUDE_CPPFLOW_CPPFLOW_H_
