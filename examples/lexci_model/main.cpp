#include <iostream>
#include <fstream>
using std::ofstream;

#include "cppflow/ops.h"
#include "cppflow/model.h"


int main() {
    std::cout << "============Reading greyscale input file from JPEG" << std::endl;
    auto input = cppflow::decode_jpeg(cppflow::read_file(std::string(IMAGE_PATH)), 1);
    input = cppflow::cast(input, TF_UINT8, TF_FLOAT);
    input = cppflow::expand_dims(input, 0);
    
    std::cout << "============Reading model" << std::endl;
    cppflow::model model(std::string(MODEL_PATH));

    std::cout << "=========Running model" << std::endl;
    auto output = model({{"serving_default_input0:0", input}}, {"StatefulPartitionedCall:0"});

    std::cout << "=========Get results" << std::endl;
    auto values = output[0].get_data<float>();

    std::cout << "=========Save to file" << std::endl;
    ofstream outdata;
    outdata.open("./lexci_model_output.dat");
    for (auto v : values) {
        outdata << v << std::endl;
    }
    outdata.close();

    return 0;
}
