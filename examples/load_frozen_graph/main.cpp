#include <iostream>

#include "cppflow/ops.h"
#include "cppflow/model.h"


int main() {

    auto input = cppflow::fill({10, 5}, 1.0f);
    std::cout << "start" << std::endl;
    cppflow::model model("../model.pb", cppflow::model::FROZEN_GRAPH);
    auto output = model({{"x:0", input}}, {{"Identity:0"}})[0];
    
    std::cout << output << std::endl;

    auto values = output.get_data<float>();

    for (auto v : values) {
        std::cout << v << std::endl;
    }
    return 0;
}
