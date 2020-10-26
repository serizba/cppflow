#include <iostream>
#include <typeinfo>
/*IF USING OPENCV
#include <opencv2/opencv.hpp>
*/
#include <algorithm>
#include <iterator>
#include "cppflow/cppflow.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


int main() {

    auto input = cppflow::decode_jpeg(cppflow::read_file(std::string("../small.jpg")), 0, 1, true, false, 1.0, std::string("INTEGER_ACCURATE"));
    input = cppflow::cast(input, TF_UINT8, TF_FLOAT);
    input = cppflow::expand_dims(input, 0);
    int upscale_factor = 4;

    std::cout << input << std::endl;


    auto shape = input.shape().get_data<int>();
    int num_row = shape[1];
    int num_col = shape[2];
    int num_channel = shape[3];

    cppflow::model model("../model");
    auto output = model(input);
    output = cppflow::cast(output, TF_FLOAT, TF_UINT8);
    //added
    auto data = output.get_data<uint8_t>();

    unsigned char* result = data.data();
    stbi_write_png("../result.png", num_col*upscale_factor, num_row*upscale_factor, num_channel, result, num_col*upscale_factor * num_channel);

    /*IF USING OPENCV
    std::vector<uint8_t> sr_img;
    sr_img.assign(data.begin(), data.end());

    std::cout << sr_img.size()*sizeof(uint8_t) << std::endl;

    cv::Mat save = cv::Mat(num_row*upscale_factor, num_col*upscale_factor, CV_8UC3);

    memcpy(save.data, sr_img.data(), sr_img.size()*sizeof(uint8_t));

    cv::cvtColor(save, save, cv::COLOR_RGB2BGR);
    cv::imwrite("../result.png", save);
    */

    return 0;
}
//end
