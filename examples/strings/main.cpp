// MIT License
//
// Copyright (c) 2022 Sergio Izquierdo
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
 *  @file       main.cpp
 *  @author     Sergio Izquierdo
 *  @date       @showdate "%B %d, %Y" 2022-09-23
 */

// CppFlow headers
#include <cppflow/cppflow.h>

// C++ headers
#include <iostream>

int main() {
  std::string my_string{
      "A very nice string that is somewhat long so it is stored in the heap"};
  std::vector<std::string> my_vector{my_string, my_string, my_string};
  auto scalar_tensor = cppflow::tensor(my_string);
  auto matrix_tensor = cppflow::tensor(my_vector, {1, 3});
  auto vector_tensor = cppflow::tensor({my_string, my_string, my_string});

  std::cout << scalar_tensor << std::endl;
  std::cout << matrix_tensor << std::endl;
  std::cout << vector_tensor << std::endl;
  return 0;
}
