//
// Created by sergio on 13/05/19.
//

#ifndef CPPFLOW_TENSOR_H
#define CPPFLOW_TENSOR_H

#include <tensorflow/c/c_api.h>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <cstring>
#include "Model.h"

class Model;

class Tensor {
public:
    Tensor(const Model& model, const std::string& operation);
    ~Tensor();

    void clean();

    template<typename T> void set_data(std::vector<T> new_data);
    template<typename T> void set_data(std::vector<T> new_data, const std::vector<int64_t>& new_shape);
    template<typename T> std::vector<T> get_data();


private:
    TF_Tensor* val;
    TF_Output op;
    TF_DataType type;
    std::vector<int64_t> shape;
    std::vector<int64_t>* actual_shape;
    void* data;
    int flag;

    // Aux functions
    void error_check(bool condition, const std::string& error);
    template <typename T> static TF_DataType deduce_type();
    void deduce_shape(const Model& model);

public:
    friend class Model;
};

#endif //CPPFLOW_TENSOR_H
