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
 *  @file       ops.h
 *  @author     Jiannan Liu
 *  @author     Sergio Izquierdo
 *  @date       @showdate "%B %d, %Y" 2020-07-31
 */

#ifndef INCLUDE_CPPFLOW_OPS_H_
#define INCLUDE_CPPFLOW_OPS_H_

// C++ headers
#include <string>

// CppFlow headers
#include "cppflow/tensor.h"
#include "cppflow/raw_ops.h"

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
}  // namespace cppflow

/******************************
 *   IMPLEMENTATION DETAILS   *
 ******************************/

namespace cppflow {

// Operators

inline tensor operator+(const tensor& x, const tensor& y) {
  return add(x, y);
}

inline tensor operator-(const tensor& x, const tensor& y) {
  return sub(x, y);
}

inline tensor operator*(const tensor& x, const tensor& y) {
  return mul(x, y);
}

inline tensor operator/(const tensor& x, const tensor& y) {
  return div(x, y);
}

inline std::ostream& operator<<(std::ostream& os, const cppflow::tensor& t) {
  std::string res =  to_string(t);
  return os << res;
}


inline std::string to_string(const tensor &t) {
  auto res_tensor = string_format({t.shape(), t},
      "(tensor: shape=%s, dtype="+ to_string(t.dtype()) + ", data=\n%s)");
  auto res_tensor_h = res_tensor.get_tensor();

#ifdef TENSORFLOW_C_TF_TSTRING_H_
  // For future version TensorFlow 2.4
  // auto *t_str = reinterpret_cast<TF_TString *>(
  //     TF_TensorData(res_tensor_h.get()));
  auto *t_str = (TF_TString *)(TF_TensorData(res_tensor_h.get()));
  auto result = std::string(TF_TString_GetDataPointer(t_str),
                            TF_TString_GetSize(t_str));
#else
  const char* dst[1] = {nullptr};
  size_t dst_len[1] = {3};
  TF_StringDecode(static_cast<char*>(TF_TensorData(res_tensor_h.get())) + 8,
                  TF_TensorByteSize(res_tensor_h.get()), dst, dst_len,
                  context::get_status());
  status_check(context::get_status());
  auto result = std::string(dst[0], *dst_len);
#endif  // TENSORFLOW_C_TF_TSTRING_H_

  return result;
}

}  // namespace cppflow

#endif  // INCLUDE_CPPFLOW_OPS_H_
