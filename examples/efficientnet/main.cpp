#include <iostream>
#include <cstdlib>

#include "cppflow/cppflow.h"

void print_meminfo() {
    std::system("nvidia-smi");
}

int main() {
    // limit GPU memory usage to 30%
    std::vector<uint8_t> config{0x32,0xb,0x9,0x34,0x33,0x33,0x33,0x33,0x33,0xd3,0x3f,0x20,0x1};
    TFE_ContextOptions* options = TFE_NewContextOptions();
    TFE_ContextOptionsSetConfig(options, config.data(), config.size(), cppflow::context::get_status());
    cppflow::get_global_context() = cppflow::context(options);

    print_meminfo();

    auto input = cppflow::decode_jpeg(cppflow::read_file(std::string("../my_cat.jpg")));
    input = cppflow::cast(input, TF_UINT8, TF_FLOAT);
    input = cppflow::expand_dims(input, 0);
    cppflow::model model("../model");
    auto output = model(input);

    std::cout << "It's a tiger cat: " << cppflow::arg_max(output, 1) << std::endl;

    print_meminfo();

    return 0;
}
