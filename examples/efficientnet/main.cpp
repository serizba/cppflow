#include <iostream>

#include "cppflow/cppflow.h"


int main() {

    auto input = cppflow::decode_jpeg(cppflow::read_file(std::string(CAT_PATH)));
    input = cppflow::cast(input, TF_UINT8, TF_FLOAT);
    input = cppflow::expand_dims(input, 0);
    cppflow::model model(std::string(MODEL_PATH));
    auto output = model(input);

    std::cout << "It's a tiger cat: " << cppflow::arg_max(output, 1) << std::endl;

    return 0;
}
