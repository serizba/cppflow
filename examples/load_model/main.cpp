// MIT License
//
// Copyright (c) 2019 Sergio Izquierdo
// Copyright (c) 2019 Paul Nykiel
// Copyright (c) 2020 Afaq Sabir
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
 *  @author     Afaq Sabir
 *  @author     Florian
 *  @author     Paul Nykiel
 *  @author     Sergio Izquierdo
 *  @date       @showdate "%B %d, %Y" 2019-05-16
 */

// CppFlow headers
#include <cppflow/ops.h>
#include <cppflow/model.h>

// C++ headers
#include <iostream>

int main() {
    auto input = cppflow::fill({10, 5}, 1.0f);
    cppflow::model model(std::string(MODEL_PATH));
    auto output = model(input);

    std::cout << output << std::endl;

    auto values = output.get_data<float>();

    for (auto v : values) {
        std::cout << v << std::endl;
    }
    return 0;
}
