//
// Created by serizba on 29/6/20.
//

#ifndef CPPFLOW2_MODEL_H
#define CPPFLOW2_MODEL_H

#include <tensorflow/c/c_api.h>
#include <string>
#include <fstream>
#include <iostream>
#include <vector>

#include "context.h"
#include "tensor.h"

namespace cppflow {

    class model {
    public:
        explicit model(const std::string& filename);

        std::vector<std::string> get_operations() const;

        //std::vector<tensor> operator()(std::vector<std::tuple<std::string, tensor>> inputs, std::vector<tensor> outputs);

        // At the moment only default run with one input and one output is implemented
        tensor operator()(const tensor& input);


    private:

        TF_Graph* graph;
        TF_Session* session;
    };
}

namespace cppflow {

    model::model(const std::string &filename) {
        this->graph = TF_NewGraph();

        // Create the session.
        TF_SessionOptions* session_options = TF_NewSessionOptions();
        TF_Buffer* run_options = TF_NewBufferFromString("", 0);
        TF_Buffer* meta_graph = TF_NewBuffer();

        int tag_len = 1;
        const char* tag = "serve";
        this->session = TF_LoadSessionFromSavedModel(session_options, run_options, filename.c_str(), &tag, tag_len, graph, meta_graph, context::get_status());
        TF_DeleteSessionOptions(session_options);
        TF_DeleteBuffer(run_options);
        //TF_DeleteBuffer(meta_graph);

        status_check(context::get_status());
    }

    std::vector<std::string> model::get_operations() const {
        std::vector<std::string> result;
        size_t pos = 0;
        TF_Operation* oper;

        // Iterate through the operations of a graph
        while ((oper = TF_GraphNextOperation(this->graph, &pos)) != nullptr) {
            result.emplace_back(TF_OperationName(oper));
        }
        return result;
    }

    tensor model::operator()(const tensor& input) {
        auto inputs = new TF_Output[1];
        inputs[0].oper = TF_GraphOperationByName(this->graph, "serving_default_input_1");
        inputs[0].index = 0;

        TF_Output op2[1];
        op2[0].oper = TF_GraphOperationByName(this->graph, "StatefulPartitionedCall");
        op2[0].index = 0;


        //********* Allocate data for inputs & outputs
        auto inp_tensor = TFE_TensorHandleResolve(input.tfe_handle.get(), context::get_status());
        status_check(context::get_status());


        TF_Tensor* inpvals[1] = {inp_tensor};
        TF_Tensor* outvals[1] = {nullptr};


        TF_SessionRun(this->session, NULL, inputs, inpvals, 1, op2, outvals, 1, NULL, 0,NULL , context::get_status());
        status_check(context::get_status());

        return tensor(outvals[0]);
    }
}

#endif //CPPFLOW2_MODEL_H
