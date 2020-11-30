//
// Created by serizba on 29/6/20.
//

#ifndef CPPFLOW2_MODEL_H
#define CPPFLOW2_MODEL_H

#include <tensorflow/c/c_api.h>
#include <string>
#include <fstream>
#include <iostream>
#include <tuple>
#include <vector>

#include "context.h"
#include "defer.h"
#include "tensor.h"

namespace cppflow {

    class model {
    public:
        explicit model(const std::string& filename);

        std::vector<std::string> get_operations() const;

        std::vector<tensor> operator()(std::vector<std::tuple<std::string, tensor>> inputs, std::vector<std::string> outputs);
        tensor operator()(const tensor& input);

        ~model() = default;
        model(const model &model) = default;
        model(model &&model) = default;
        model &operator=(const model &other) = default;
        model &operator=(model &&other) = default;

    private:

        std::shared_ptr<TF_Graph> graph;
        std::shared_ptr<TF_Session> session;
    };
}
#endif //CPPFLOW2_MODEL_H
