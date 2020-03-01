//
// Created by sergio on 16/05/19.
//

#include "../../include/Model.h"
#include "../../include/Tensor.h"

#include <numeric>
#include <iomanip>

int main() {
    Model model("../model.pb");
    model.init();

    Tensor input_a{model, "input_a"};
    Tensor input_b{model, "input_b"};
    Tensor output{model, "result"};

    std::vector<float> data(100);
    std::iota(data.begin(), data.end(), 0);

    input_a.set_data(data);
    input_b.set_data(data);

    model.run({&input_a, &input_b}, output);
    for (float f : output.get_data<float>()) {
        std::cout << f << " ";
    }
    std::cout << std::endl;

}
