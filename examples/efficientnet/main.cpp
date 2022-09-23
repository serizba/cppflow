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
 *  @brief      Run EfficientNet on a cat image as an example.
 *  @details    Run an EfficientNet model on a cat image and print the result. 
 *              The EfficientNet model should be downloaded running create_model.py
 *  @author     Florian
 *  @author     Sergio Izquierdo
 *  @date       @showdate "%B %d, %Y" 2020-09-16
 */

// CppFlow headers
#include <cppflow/cppflow.h>

// C++ headers
#include <iostream>

int main() {
    auto input = cppflow::decode_jpeg(
        cppflow::read_file(std::string(CAT_PATH)));
    input = cppflow::cast(input, TF_UINT8, TF_FLOAT);
    input = cppflow::expand_dims(input, 0);
    cppflow::model model(std::string(MODEL_PATH));
    auto output = model(input);

    std::cout << "It's a tiger cat: " << cppflow::arg_max(output, 1)
              << std::endl;

    return 0;
}
