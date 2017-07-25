//
// Created by Jarlene on 2017/7/24.
//
#include <opencv2/opencv.hpp>
#include <matrix/include/utils/Logger.h>
using namespace cv;
using namespace matrix;
int main() {

    const std::string name = "/Users/jarlene/Desktop/image.png";

    Mat mat = cv::imread(name);

    Logger::Global()->Info("the image size is %d X %d channel is %d \n", mat.cols, mat.rows, mat.channels());

    return 0;
}