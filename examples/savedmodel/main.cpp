#include <iostream>
#include <fstream>
#include <exception>
#include <numeric>
#include <vector>

#include "../../include/SavedModel.h"

/*
The given SavedModel SignatureDef contains the following input(s):
  inputs['input_1'] tensor_info:
      dtype: DT_FLOAT
      shape: (-1, 224, 224, 3)
      name: serving_default_input_1:0
The given SavedModel SignatureDef contains the following output(s):
  outputs['act_softmax'] tensor_info:
      dtype: DT_FLOAT
      shape: (-1, 1000)
      name: StatefulPartitionedCall:0
Method name is: tensorflow/serving/predict
*/

#define IMAGE_SIZE 224
#define OUTPUT_SIZE 1000

void read_testcase(std::vector<float>& x, std::vector<float>& results) {
    std::ifstream f("testcase");
    if(!f.is_open()) {
        throw std::runtime_error{"Failed to open testcase"};
    }

    x.resize(IMAGE_SIZE*IMAGE_SIZE*3);
    results.resize(OUTPUT_SIZE);

    for(int i = 0; i < IMAGE_SIZE*IMAGE_SIZE*3; i++) {
        f >> x[i];
    }

    for(int i = 0; i < OUTPUT_SIZE; i++) {
        f >> results[i];
    }
}

int main() {
    SavedModel model("mobilenet/1/");
    
    std::cout << "get_operations():" << std::endl;
    for(auto s: model.get_operations()) {
        std::cout << s << std::endl;
    }

    Tensor input{model, "serving_default_input_1"};
    Tensor output{model, "StatefulPartitionedCall"};

    std::vector<float> x;
    std::vector<float> y;
    read_testcase(x, y);

    input.set_data(x);

    model.run(input, output);
    auto y_new = output.get_data<float>();
    for(float f : y_new) {
        std::cout << f << " ";
    }
    std::cout << std::endl;

    bool precision_test_pass = true;
    for(int i = 0; i < y_new.size(); i++) {
        float err = std::abs(y_new[i]/y[i]-1);
        if(err > 1.e-4) {
            precision_test_pass = false;
            std::cout << "Error " << err << " at i=" << i
            << " is too large. y=" << y[i]
            << " y_new=" << y_new[i] << std::endl;
        }
    }

    if(precision_test_pass) {
        std::cout << "test pass" << std::endl;
    }

    return 0;
}
