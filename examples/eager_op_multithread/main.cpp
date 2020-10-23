#include <thread>
#include <cmath>
#include <vector>
#include <iostream>

#include "cppflow/cppflow.h"

constexpr size_t num_iter = 10240;
constexpr size_t num_threads = 32;

void test1(float input1, float input2, float input3) {
    float target = (input1+input2)*input3;
    for(size_t i = 0; i < num_iter; i++) {
        cppflow::tensor t1(input1), t2(input2), t3(input3);
        auto result = (t1+t2)*t3;
        float result_value = result.get_data<float>()[0];
        if(std::abs(target/result_value-1.0f) > 1e-6) {
            std::cout << "error: result_value=" << result_value << ", target=" << target << std::endl;
        }
    }
}

int main() {
    std::vector<std::thread> threads;
    for(size_t i = 0; i < num_threads; i++) {
        if(i % 2 == 0) {
            std::thread t1(test1, 3, 10, 100);
            threads.push_back(std::move(t1));
        } else {
            std::thread t2(test1, 130, 10, 100);
            threads.push_back(std::move(t2));
        }
    }

    for(auto& t: threads) {
        t.join();
    }

    return 0;
}
