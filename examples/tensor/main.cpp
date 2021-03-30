#include <cmath>
#include <vector>
#include <iostream>
#include <stdexcept>

#include "cppflow/cppflow.h"

bool float_equal(const float f1, const float f2) {
    return std::abs(f1/f2-1.0f) < 1e-6;
}

void test1(const bool is_cpu) {
    std::cout << "test1 starts: is_cpu=" << is_cpu << std::endl;
    float target = 1.0;
    int64_t ndim = 2;
    cppflow::tensor t1;

    if(is_cpu) {
        std::vector<float> _data(ndim, target);
        t1 = cppflow::tensor(_data, {ndim});
    } else {
        t1 = cppflow::fill({ndim}, target);
    }

    std::cout << "tensor::device(true) : " << t1.device(true) << std::endl;
    std::cout << "tensor::device(false) : " << t1.device(false) << std::endl;

    auto t1_tensor = t1.get_tensor();
    auto raw_data = static_cast<float*>(TF_TensorData(t1_tensor.get()));
    float result_value = raw_data[0];
    if(float_equal(result_value, target)) {
        std::cout << "tensor::get_tensor() test1-1: pass" << std::endl;
    } else {
        std::cout << "tensor::get_tensor() test1-1: result_value=" << result_value << ", target=" << target << std::endl;
        throw std::runtime_error("tensor::get_tensor() test1-1: failed");
    }

    // IMPORTANT NOTE: CANNOT modify the returned cache
    float target2 = target + 10.0;
    raw_data[1] = target2;
    result_value = t1.get_data<float>()[0];
    float result_value2 = t1.get_data<float>()[1];
    if(float_equal(result_value, target)) {
        std::cout << "tensor::get_tensor() test1-2: pass" << std::endl;
    } else {
        std::cout << "tensor::get_tensor() test1-2: failed, result_value=" << result_value << ", target=" << target << std::endl;
        throw std::runtime_error("tensor::get_tensor() test1-2: failed");
    }
    if(float_equal(result_value2, target2)) {
        std::cout << "tensor::get_tensor() test1-3: pass" << std::endl;
    } else {
        std::cout << "The failure of test1-3 is not considered as a bug." << std::endl;
        std::cout << "tensor::get_tensor() test1-3: failed, result_value=" << result_value2 << ", target2=" << target2 << std::endl;
    }

    auto t2 = t1 + cppflow::tensor(0.f);
    std::cout << "Can NOT modify the cache!" << std::endl;
    std::cout << "t2: " << t2 << std::endl;

    auto dt = cppflow::to_string(t1.dtype());
    std::string expected_dtype{"TF_FLOAT"};
    if(dt == expected_dtype) {
        std::cout << "tensor::get_tensor() test1-4: pass" << std::endl;
    } else {
        std::cout << "tensor::get_tensor() test1-4: dtype=" << dt << ", expected_dtype=" << expected_dtype << std::endl;
        throw std::runtime_error("tensor::get_tensor() test1-4: failed");
    }

    auto shape_tensor = t1.shape();
    auto shape = shape_tensor.get_data<int64_t>()[0];
    if(shape == ndim) {
        std::cout << "tensor::get_tensor() test1-5: pass" << std::endl;
    } else {
        std::cout << "tensor::get_tensor() test1-5: shape_tensor.dtype()=" << cppflow::to_string(shape_tensor.dtype()) << std::endl;
        std::cout << "tensor::get_tensor() test1-5: shape_tensor=" << shape_tensor << std::endl;
        std::cout << "tensor::get_tensor() test1-5: shape()=" << shape << ", ndim=" << ndim << std::endl;
        throw std::runtime_error("tensor::get_tensor() test1-5: failed");
    }

    std::cout << std::endl;
}

int main() {
    test1(true);
    test1(false);

    return 0;
}
