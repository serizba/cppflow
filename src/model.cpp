//
// Created by serizba on 29/6/20.
//


#include "cppflow/model.h"


namespace cppflow {

    model::model(const std::string &filename) {
        this->graph = {TF_NewGraph(), TF_DeleteGraph};

        // Create the session.
        std::unique_ptr<TF_SessionOptions, decltype(&TF_DeleteSessionOptions)> session_options = {TF_NewSessionOptions(), TF_DeleteSessionOptions};
        std::unique_ptr<TF_Buffer, decltype(&TF_DeleteBuffer)> run_options = {TF_NewBufferFromString("", 0), TF_DeleteBuffer};
        std::unique_ptr<TF_Buffer, decltype(&TF_DeleteBuffer)> meta_graph = {TF_NewBuffer(), TF_DeleteBuffer};

        auto session_deleter = [](TF_Session* sess) {
            TF_DeleteSession(sess, context::get_status());
            status_check(context::get_status());
        };

        int tag_len = 1;
        const char* tag = "serve";
        this->session = {TF_LoadSessionFromSavedModel(session_options.get(), run_options.get(), filename.c_str(),
                                &tag, tag_len, this->graph.get(), meta_graph.get(), context::get_status()),
                         session_deleter};

        status_check(context::get_status());
    }

    std::vector<std::string> model::get_operations() const {
        std::vector<std::string> result;
        size_t pos = 0;
        TF_Operation* oper;

        // Iterate through the operations of a graph
        while ((oper = TF_GraphNextOperation(this->graph.get(), &pos)) != nullptr) {
            result.emplace_back(TF_OperationName(oper));
        }
        return result;
    }

    std::tuple<std::string, int> parse_name(const std::string& name) {
        auto idx = name.find(':');
        return (idx == -1 ? std::make_tuple(name, 0) : std::make_tuple(name.substr(0, idx), std::stoi(name.substr(idx + 1))));
    }

    std::vector<tensor> model::operator()(std::vector<std::tuple<std::string, tensor>> inputs, std::vector<std::string> outputs) {

        std::vector<TF_Output> inp_ops(inputs.size());
        std::vector<TF_Tensor*> inp_val(inputs.size());
        for (int i=0; i<inputs.size(); i++) {

            // Operations
            const auto tupleReturn = parse_name(std::get<0>(inputs[i]));
            auto op_name = std::get<0>(tupleReturn);
            auto op_idx = std::get<1>(tupleReturn);
        	
            inp_ops[i].oper = TF_GraphOperationByName(this->graph.get(), op_name.c_str());
            inp_ops[i].index = op_idx;

            if (!inp_ops[i].oper)
                throw std::runtime_error("No operation named \"" + op_name + "\" exists");

            // Values
            auto inp_tensor = TFE_TensorHandleResolve(std::get<1>(inputs[i]).tfe_handle.get(), context::get_status());
            status_check(context::get_status());
            inp_val[i] = inp_tensor;
        }

        std::vector<TF_Output> out_ops(outputs.size());
        auto out_val = std::make_unique<TF_Tensor*[]>(outputs.size());
        for (int i=0; i<outputs.size(); i++) {

            const auto tupleReturn = parse_name(outputs[i]);
            auto op_name = std::get<0>(tupleReturn);
            auto op_idx = std::get<1>(tupleReturn);
        	
            out_ops[i].oper = TF_GraphOperationByName(this->graph.get(), op_name.c_str());
            out_ops[i].index = op_idx;

            if (!out_ops[i].oper)
                throw std::runtime_error("No operation named \"" + op_name + "\" exists");

        }

        TF_SessionRun(this->session.get(), NULL,
                inp_ops.data(), inp_val.data(), inputs.size(),
                out_ops.data(), out_val.get(), outputs.size(),
                NULL, 0,NULL , context::get_status());
        status_check(context::get_status());

        std::vector<tensor> result;
        result.reserve(outputs.size());
        for (int i=0; i<outputs.size(); i++) {
            result.emplace_back(tensor(out_val[i]));
        }

        return result;
    }

    tensor model::operator()(const tensor& input) {
        return (*this)({{"serving_default_input_1", input}}, {"StatefulPartitionedCall"})[0];
    }
}
