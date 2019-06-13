//
// Created by sergio on 12/05/19.
//

#ifndef CPPFLOW_MODEL_H
#define CPPFLOW_MODEL_H

#include <tensorflow/c/c_api.h>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <tuple>
#include "Tensor.h"

class Tensor;

class Model {
public:
    explicit Model(const std::string&);
    ~Model();

    void init();
    void restore(const std::string& ckpt);
    void save(const std::string& ckpt);
    std::vector<std::string> get_operations() const;

    // Original Run
    void run(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs);

    // Run with references
    void run(Tensor& input, const std::vector<Tensor*>& outputs);
    void run(const std::vector<Tensor*>& inputs, Tensor& output);
    void run(Tensor& input, Tensor& output);

    // Run with pointers
    void run(Tensor* input, const std::vector<Tensor*>& outputs);
    void run(const std::vector<Tensor*>& inputs, Tensor* output);
    void run(Tensor* input, Tensor* output);

private:
    TF_Graph* graph;
    TF_Session* session;
    TF_Status* status;

    // Read a file from a string
    TF_Buffer* read(const std::string&);

    bool status_check(bool throw_exc) const;
    void error_check(bool condition, const std::string &error) const;

public:
    friend class Tensor;
};


#endif //CPPFLOW_MODEL_H
