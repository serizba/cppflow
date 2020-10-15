#include <iostream>

#include "cppflow/ops.h"
#include "cppflow/model.h"

int main() {

    auto input_1 = cppflow::fill({10, 5}, 1.0f);
    auto input_2 = cppflow::fill({10, 5}, -1.0f);
    cppflow::model model("../model");

    auto output = model({{"serving_default_my_input_1:0", input_1}, {"serving_default_my_input_2:0", input_2}}, {"StatefulPartitionedCall:0", "StatefulPartitionedCall:1"});

    std::cout << "output_1: " << output[0] << std::endl;
    std::cout << "output_2: " << output[1] << std::endl;
    return 0;
}
