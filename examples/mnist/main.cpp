//
// Created by sergio on 12/05/19.
//

#include "../../include/Model.h"
#include "../../include/Tensor.h"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <iterator>


int main() {

    // Create model
    Model m("../model.pb");
    m.restore("../checkpoint/train.ckpt");

    // Create Tensors
    Tensor input(m, "input");
    Tensor prediction(m, "prediction");

    // Read image
    for (int i=0; i<10; i++) {
        cv::Mat img, scaled;

        // Read image
        img = cv::imread("../images/"+std::to_string(i)+".png");

        // Scale image to range 0-1
        img.convertTo(scaled, CV_64F, 1.f/255);

        // Put image in vector
        std::vector<double> img_data;
        img_data.assign(scaled.begin<double>(), scaled.end<double>());

        // Feed data to input tensor
        input.set_data(img_data);

        // Run and show predictions
        m.run(input, prediction);

        // Get tensor with predictions
        auto result = prediction.Tensor::get_data<double>();

        // Maximum prob
        auto max_result = std::max_element(result.begin(), result.end());

        // Print result
        std::cout << "Real label: " << i << ", predicted: " << std::distance(result.begin(), max_result) 
                  << ", Probability: " << (*max_result) << std::endl;

    }
}
