#include <iostream>
#include <fstream>
#include <exception>
#include <cmath>
#include <vector>

#include "../../include/cppflow/model_v2.h"

/* saved_model_cli show --dir model --all
MetaGraphDef with tag-set: 'serve' contains the following SignatureDefs:

signature_def['__saved_model_init_op']:
  The given SavedModel SignatureDef contains the following input(s):
  The given SavedModel SignatureDef contains the following output(s):
    outputs['__saved_model_init_op'] tensor_info:
        dtype: DT_INVALID
        shape: unknown_rank
        name: NoOp
  Method name is: 

signature_def['serving_default']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['val1'] tensor_info:
        dtype: DT_FLOAT
        shape: ()
        name: serving_default_val1:0
    inputs['val2'] tensor_info:
        dtype: DT_FLOAT
        shape: ()
        name: serving_default_val2:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['result1'] tensor_info:
        dtype: DT_FLOAT
        shape: ()
        name: StatefulPartitionedCall:0
    outputs['result2'] tensor_info:
        dtype: DT_FLOAT
        shape: ()
        name: StatefulPartitionedCall:1
  Method name is: tensorflow/serving/predict

Defined Functions:
  Function Name: '__call__'
    Option #1
      Callable with:
        Argument #1
          val1: TensorSpec(shape=(), dtype=tf.float32, name='val1')
        Argument #2
          val2: TensorSpec(shape=(), dtype=tf.float32, name='val2')
*/

bool check_outputs(std::unordered_map<std::string, cppflow::tensor> y, std::unordered_map<std::string, std::vector<float>> target) {
    if(y.size() != target.size()) {
        throw std::runtime_error{"check_outputs(): Sizes of y and target mismatch."};
    }
    
    double error{0};
    std::cout << "model outputs: " << std::endl;
    for(auto& [key, val]: y) {
        std::cout << "name: " << key << std::endl;
        auto output_data = val.get_data<float>();
        auto& target_data = target.at(key);
        for(size_t j = 0; j < output_data.size(); j++) {
            error += std::abs(static_cast<double>(output_data[j])/static_cast<double>(target_data[j])-1);
            std::cout << output_data[j] << " " << std::endl;
        }
        std::cout << "end name: " << key << std::endl;
    }
    std::cout << "end model outputs" << std::endl;
    
    if(error < 1e-6) {
        std::cout << "test pass" << std::endl << std::endl;
        return true;
    } else {
        std::cout << "test failed" << std::endl << std::endl;
        return false;
    }
}

bool check_outputs(std::vector<cppflow::tensor> y, std::vector<std::vector<float>> target) {
    if(y.size() != target.size()) {
        throw std::runtime_error{"check_outputs(): Sizes of y and target mismatch."};
    }
    
    double error{0};
    std::cout << "model outputs: " << std::endl;
    for(size_t i = 0; i < y.size(); i++) {
        std::cout << "output idx=" << i << std::endl;
        auto output_data = y[i].get_data<float>();
        for(size_t j = 0; j < output_data.size(); j++) {
            error += std::abs(static_cast<double>(output_data[j])/static_cast<double>(target[i][j])-1);
            std::cout << output_data[j] << " " << std::endl;
        }
        std::cout << "end output idx=" << i << std::endl;
    }
    std::cout << "end model outputs" << std::endl;
    
    if(error < 1e-6) {
        std::cout << "test pass" << std::endl << std::endl;
        return true;
    } else {
        std::cout << "test failed" << std::endl << std::endl;
        return false;
    }
}

void test_add_one(const std::string& tag = "") {
    std::cout << "Loading model" << std::endl;
    
    std::vector<std::string> tags;
    if(!tag.empty()) {
        std::cout << "tag: " << tag << std::endl;
        tags.push_back(tag);
    }
    cppflow::model_v2 model("../model/", tags);
    
    auto [inputs_name, outupts_name] = model.get_signature_def_function_operations();
    
    std::cout << "get_signature_def_function_operations " << std::endl;
    std::cout << "input tensor name: " << std::endl;
    for(auto& s: inputs_name) {
        std::cout << s << std::endl;
    }
    std::cout << "end input tensor name" << std::endl << std::endl;
    
    std::cout << "output tensor name: " << std::endl;
    for(auto& s: outupts_name) {
        std::cout << s << std::endl;
    }
    std::cout << "end output tensor name" << std::endl << std::endl;

    cppflow::tensor val1(3.0f);
    cppflow::tensor val2(10.0f);
    std::vector<float> result1{24.0};
    std::vector<float> result2{44.0};
    
    std::vector<cppflow::tensor> inputs_list{val1, val2};
    std::vector<std::vector<float>> outputs_list{result1, result2};
    std::unordered_map<std::string, cppflow::tensor> inputs_dict{{"val1", val1}, {"val2", val2}};
    std::unordered_map<std::string, std::vector<float>> outputs_dict{{"result1", result1}, {"result2", result2}};
    
    bool success{true};

    {
        std::cout << "test run_signature_def_function dict" << std::endl;
        auto y = model.run_signature_def_function(inputs_dict);
        success = check_outputs(y, outputs_dict) && success;
    }

    /*{
        std::cout << "test run_concrete_function" << std::endl;
        auto y = model.run_concrete_function(inputs_dict);
        success = check_outputs(y, outputs_dict) && success;
    }*/

    {
        std::cout << "test operator() with run_signature_def_function list" << std::endl;
        auto y = model(inputs_list, "serving_default");
        success = check_outputs(y, outputs_list) && success;
    }

    {
        std::cout << "test operator() with run_signature_def_function dict" << std::endl;
        auto y = model(inputs_dict, "serving_default");
        success = check_outputs(y, outputs_dict) && success;
    }

    /*{
        std::cout << "test operator() with run_concrete_function list" << std::endl;
        auto y = model(inputs_list, "__call__");
        success = check_outputs(y, outputs_list) && success;
    }*/

    /*{
        std::cout << "test operator() with run_concrete_function list" << std::endl;
        auto y = model(inputs_dict, "__call__");
        success = check_outputs(y, outputs_dict) && success;
    }*/
    
    if(success) {
        std::cout << "All tests passed" << std::endl << std::endl;
    } else {
        std::cout << "Some tests failed" << std::endl << std::endl;
    }
}

int main() {
    test_add_one();
    test_add_one("serve");
    return 0;
}
