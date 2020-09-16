#include <iostream>

#include "cppflow/ops.h"
#include "cppflow/model.h"


int main() {

    auto input = cppflow::decode_jpeg(cppflow::read_file(std::string("../my_cat.jpg")));
    input = cppflow::cast(input, TF_UINT8, TF_FLOAT);
    input = cppflow::expand_dims(input, 0);
    cppflow::model m("../model");
    auto output = m.run(input);

    std::cout << "It's a tiger cat: " << cppflow::arg_max(output, 1) << std::endl;
    
    return 0;
}
