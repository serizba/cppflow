#include <iostream>

#include "cppflow/ops.h"
#include "cppflow/model.h"


int main() {

    auto input = cppflow::decode_jpeg(cppflow::read_file(std::string(IMAGE_PATH)));
    cppflow::model model(std::string(MODEL_PATH));
    auto output = model(input);

    std::cout << output << std::endl;

    /*auto values = output.get_data<float>();

    for (auto v : values) {
        std::cout << v << std::endl;
    }*/

    return 0;
}
