// MIT License
//
// Copyright (c) 2020 Sergio Izquierdo
// Copyright (c) 2021 Florian
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
 *  @brief      Loads a saved multi input/output model and runs it with dummy data
 *  @details    Loads a simple Keras model with multiple inputs and outputs saved 
 *              in saved model format and runs it with dummy data
 *  @author     Florian
 *  @author     Sergio Izquierdo
 *  @date       @showdate "%B %d, %Y" 2020-10-15
 */

// CppFlow headers
#include <cppflow/ops.h>
#include <cppflow/model.h>

// C++ headers
#include <iostream>

int main() {
    auto input_1 = cppflow::fill({10, 5}, 1.0f);
    auto input_2 = cppflow::fill({10, 5}, -1.0f);
    cppflow::model model(std::string(MODEL_PATH));

    auto output = model({{"serving_default_my_input_1:0", input_1},
                         {"serving_default_my_input_2:0", input_2}},
                        {"StatefulPartitionedCall:0",
                         "StatefulPartitionedCall:1"});

    std::cout << "output_1: " << output[0] << std::endl;
    std::cout << "output_2: " << output[1] << std::endl;
    return 0;
}
