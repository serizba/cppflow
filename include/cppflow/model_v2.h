/*
MIT License

Copyright (c) 2020 Jiannan Liu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#ifndef CPPFLOW2_MODEL_V2_H
#define CPPFLOW2_MODEL_V2_H

#include <tensorflow/c/c_api.h>
#include <tensorflow/c/experimental/saved_model/public/c_saved_model_api.h>
#include <string>
#include <iostream>
#include <vector>
#include <exception>
#include <memory>
#include <cstdlib>
#include <unordered_map>

#include "context.h"
#include "tensor.h"

namespace cppflow {

    class model_v2 {
    public:
        enum function_type { unknown, concrete, signature_def };
        
        explicit model_v2(const std::string& model_dir, const std::vector<std::string>& tags = {});
        
        ~model_v2();

        std::tuple<std::vector<std::string>, std::vector<std::string>>
        get_signature_def_function_operations(const std::string& function_name = "serving_default");
        
        std::vector<tensor> run_signature_def_function(
            const std::vector<tensor>& inputs,
            const std::string& function_name = "serving_default"
        );
        
        std::unordered_map<std::string, tensor> run_signature_def_function(
            const std::unordered_map<std::string, tensor>& inputs,
            const std::string& function_name = "serving_default"
        );
        
        std::vector<tensor> run_concrete_function(
            const std::vector<tensor>& inputs,
            const std::string& function_name = "__call__"
        );
        
        std::vector<tensor> operator()(
            const std::vector<tensor>& inputs,
            const std::string& function_name = "serving_default",
            model_v2::function_type func_type = model_v2::unknown
        );
        
        std::unordered_map<std::string, tensor> operator()(
            const std::unordered_map<std::string, tensor>& inputs,
            const std::string& function_name = "serving_default",
            model_v2::function_type func_type = model_v2::unknown
        );

    protected:
        TF_SavedModel* tf_savedmodel{nullptr};
    };
}

namespace cppflow {

    inline model_v2::model_v2(const std::string& model_dir, const std::vector<std::string>& tags) {
        if(tags.size() == 0) {
            tf_savedmodel = TF_LoadSavedModel(model_dir.c_str(), context::get_context(), context::get_status());
        } else {
            std::vector<const char*> c_tags;
            c_tags.reserve(tags.size());
            for(auto& tag: tags) {
                c_tags.push_back(tag.c_str());
            }
            int len_tags = c_tags.size();
            tf_savedmodel = TF_LoadSavedModelWithTags(model_dir.c_str(), context::get_context(), c_tags.data(), len_tags, context::get_status());
        }

        status_check(context::get_status());
    }

    inline model_v2::~model_v2() {
        TF_DeleteSavedModel(tf_savedmodel);
    }

    inline std::tuple<std::vector<std::string>, std::vector<std::string>>
    model_v2::get_signature_def_function_operations(const std::string& function_name) {
        TF_SignatureDefFunction* f = TF_GetSavedModelSignatureDefFunction(tf_savedmodel, function_name.c_str(), context::get_status());
        status_check(context::get_status());
        
        TF_SignatureDefFunctionMetadata* metadata = TF_SignatureDefFunctionGetMetadata(f);
        status_check(context::get_status());
        
        const TF_SignatureDefParamList* args = TF_SignatureDefFunctionMetadataArgs(metadata);
        const TF_SignatureDefParamList* returns = TF_SignatureDefFunctionMetadataReturns(metadata);
        const size_t num_inputs = TF_SignatureDefParamListSize(args);
        const size_t num_outputs = TF_SignatureDefParamListSize(returns);
        
        std::vector<std::string> inputs_name, outputs_name;
        for(size_t i = 0; i < num_inputs; i++) {
            const TF_SignatureDefParam* param = TF_SignatureDefParamListGet(args, i);
            inputs_name.emplace_back(TF_SignatureDefParamName(param));
        }
        for(size_t i = 0; i < num_outputs; i++) {
            const TF_SignatureDefParam* param = TF_SignatureDefParamListGet(returns, i);
            outputs_name.emplace_back(TF_SignatureDefParamName(param));
        }
        
        return std::tuple(inputs_name, outputs_name);
    }

    inline std::vector<tensor> model_v2::run_signature_def_function(
        const std::vector<tensor>& inputs,
        const std::string& function_name
    ) {
        TF_SignatureDefFunction* f = TF_GetSavedModelSignatureDefFunction(tf_savedmodel, function_name.c_str(), context::get_status());
        status_check(context::get_status());
        
        TF_SignatureDefFunctionMetadata* metadata = TF_SignatureDefFunctionGetMetadata(f);
        status_check(context::get_status());
        
        const TF_SignatureDefParamList* args = TF_SignatureDefFunctionMetadataArgs(metadata);
        const TF_SignatureDefParamList* returns = TF_SignatureDefFunctionMetadataReturns(metadata);
        const size_t num_inputs = TF_SignatureDefParamListSize(args);
        const size_t num_outputs = TF_SignatureDefParamListSize(returns);
        
        if(num_inputs != inputs.size()) {
            throw std::runtime_error{"model_v2::run_signature_def_function(): Number of input tensors does not match."};
        }
        
        std::vector<TFE_TensorHandle*> inp;
        inp.reserve(num_inputs);
        for(auto& t: inputs) {
            inp.push_back(t.tfe_handle.get());
        }
        
        auto f_op = std::unique_ptr<TFE_Op, decltype(&TFE_DeleteOp)>(TF_SignatureDefFunctionMakeCallOp(f, inp.data(), inp.size(), context::get_status()), &TFE_DeleteOp);
        status_check(context::get_status());
        
        std::vector<TFE_TensorHandle*> out{num_outputs, nullptr};
        int num_retvals = num_outputs;
        TFE_Execute(f_op.get(), out.data(), &num_retvals, context::get_status());
        status_check(context::get_status());
        
        std::vector<tensor> outputs;
        outputs.reserve(num_outputs);
        for(auto& t: out) {
            outputs.emplace_back(t);
        }
        return outputs;
    }

    inline std::unordered_map<std::string, tensor> model_v2::run_signature_def_function(
        const std::unordered_map<std::string, tensor>& inputs,
        const std::string& function_name
    ) {
        TF_SignatureDefFunction* f = TF_GetSavedModelSignatureDefFunction(tf_savedmodel, function_name.c_str(), context::get_status());
        status_check(context::get_status());
        
        TF_SignatureDefFunctionMetadata* metadata = TF_SignatureDefFunctionGetMetadata(f);
        status_check(context::get_status());
        
        const TF_SignatureDefParamList* args = TF_SignatureDefFunctionMetadataArgs(metadata);
        const TF_SignatureDefParamList* returns = TF_SignatureDefFunctionMetadataReturns(metadata);
        const size_t num_inputs = TF_SignatureDefParamListSize(args);
        const size_t num_outputs = TF_SignatureDefParamListSize(returns);
        
        if(num_inputs != inputs.size()) {
            throw std::runtime_error{"model_v2::run_signature_def_function(): Number of input tensors does not match."};
        }
        
        std::vector<TFE_TensorHandle*> inp{num_inputs, nullptr};
        
        for(size_t i = 0; i < num_inputs; i++) {
            const TF_SignatureDefParam* param = TF_SignatureDefParamListGet(args, i);
            const std::string op_name = TF_SignatureDefParamName(param);
            inp[i] = inputs.at(op_name).tfe_handle.get();
        }
        
        auto f_op = std::unique_ptr<TFE_Op, decltype(&TFE_DeleteOp)>(TF_SignatureDefFunctionMakeCallOp(f, inp.data(), inp.size(), context::get_status()), &TFE_DeleteOp);
        status_check(context::get_status());
        
        std::vector<TFE_TensorHandle*> out{num_outputs, nullptr};
        int num_retvals = num_outputs;
        TFE_Execute(f_op.get(), out.data(), &num_retvals, context::get_status());
        status_check(context::get_status());
        
        std::unordered_map<std::string, tensor> outputs;
        outputs.reserve(num_outputs);
        for(size_t i = 0; i < num_outputs; i++) {
            const TF_SignatureDefParam* param = TF_SignatureDefParamListGet(returns, i);
            outputs.emplace(TF_SignatureDefParamName(param), out[i]);
        }
        return outputs;
    }
    
    // TODO Need num_outputs because there is no FunctionMetadata API yet.
    inline std::vector<tensor> model_v2::run_concrete_function(
        const std::vector<tensor>& inputs,
        const std::string& function_name
    ) {
        TF_ConcreteFunction* f = TF_GetSavedModelConcreteFunction(tf_savedmodel, function_name.c_str(), context::get_status());
        status_check(context::get_status());
        
        // TODO check number of inputs when FunctionMetadata API is available
        const size_t num_inputs = inputs.size();
        // TODO determine num_outputs using FunctionMetadata API
        const size_t num_outputs = 1;
        
        std::vector<TFE_TensorHandle*> inp;
        inp.reserve(num_inputs);
        for(auto& t: inputs) {
            inp.push_back(t.tfe_handle.get());
        }
        
        // TODO This function will be removed in the future
        auto f_op = std::unique_ptr<TFE_Op, decltype(&TFE_DeleteOp)>(TF_ConcreteFunctionMakeCallOp(f, inp.data(), inputs.size(), context::get_status()), &TFE_DeleteOp);
        status_check(context::get_status());
        
        std::vector<TFE_TensorHandle*> out{num_outputs, nullptr};
        int num_retvals = num_outputs;
        TFE_Execute(f_op.get(), out.data(), &num_retvals, context::get_status());
        status_check(context::get_status());
        
        std::vector<tensor> outputs;
        outputs.reserve(num_outputs);
        for(auto& t: out) {
            outputs.emplace_back(t);
        }
        return outputs;
    }

    inline std::vector<tensor> model_v2::operator()(
        const std::vector<tensor>& inputs,
        const std::string& function_name,
        const model_v2::function_type func_type
    ) {
        model_v2::function_type func_type_infer = func_type;
        if(func_type_infer == model_v2::unknown) {
            // infer function_type
            if(function_name == "__call__") {
                func_type_infer = model_v2::concrete;
            } else if (function_name == "serving_default") {
                func_type_infer = model_v2::signature_def;
            } else {
                throw std::runtime_error{"model_v2::operator(): function type is unknown"};
            }
        }
        
        std::vector<tensor> outputs;
        if(func_type_infer == model_v2::concrete) {
            outputs = run_concrete_function(inputs, function_name);
        }
        
        if(func_type_infer == model_v2::signature_def) {
            outputs = run_signature_def_function(inputs, function_name);
        }
        return outputs;
    }

    inline std::unordered_map<std::string, tensor> model_v2::operator()(
        const std::unordered_map<std::string, tensor>& inputs,
        const std::string& function_name,
        const model_v2::function_type func_type
    ) {
        model_v2::function_type func_type_infer = func_type;
        if(func_type_infer == model_v2::unknown) {
            // infer function_type
            if(function_name == "__call__") {
                func_type_infer = model_v2::concrete;
            } else if (function_name == "serving_default") {
                func_type_infer = model_v2::signature_def;
            } else {
                throw std::runtime_error{"model_v2::operator(): function type is unknown"};
            }
        }
        
        std::unordered_map<std::string, tensor> outputs;
        if(func_type_infer == model_v2::concrete) {
            // TODO enable this after FunctionMetadata API is available
            throw std::runtime_error{"model_v2::operator(): run_concrete_function is unimplemented"};
            //outputs = run_concrete_function(inputs, function_name);
        }
        
        if(func_type_infer == model_v2::signature_def) {
            outputs = run_signature_def_function(inputs, function_name);
        }
        return outputs;
    }
}

#endif //CPPFLOW2_MODEL_V2_H
