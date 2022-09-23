// MIT License
//
// Copyright (c) 2020 Jiannan Liu
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
 *  @brief      Test the behavior of cppflow with multiple threads
 *  @details    Test the behavior of cppflow with multiple threads
 *  @author     Jiannan Liu
 *  @author     Sergio Izquierdo
 *  @date       @showdate "%B %d, %Y" 2020-10-26
 */

// CppFlow headers
#include <cppflow/cppflow.h>

// C++ headers
#include <thread>
#include <cmath>
#include <vector>
#include <iostream>

constexpr size_t num_iter = 10240;
constexpr size_t num_threads = 32;

void test1(float input1, float input2, float input3) {
    float target = (input1+input2)*input3;
    for (size_t i = 0; i < num_iter; i++) {
        cppflow::tensor t1(input1), t2(input2), t3(input3);
        auto result = (t1+t2)*t3;
        float result_value = result.get_data<float>()[0];
        if (std::abs(target/result_value-1.0f) > 1e-6) {
            std::cout << "error: result_value=" << result_value
                      << ", target=" << target << std::endl;
        }
    }
}

int main() {
    std::vector<std::thread> threads;
    for (size_t i = 0; i < num_threads; i++) {
        if (i % 2 == 0) {
            std::thread t1(test1, 3, 10, 100);
            threads.push_back(std::move(t1));
        } else {
            std::thread t2(test1, 130, 10, 100);
            threads.push_back(std::move(t2));
        }
    }

    for (auto& t : threads) {
        t.join();
    }

    return 0;
}
