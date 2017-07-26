//
// Created by Jarlene on 2017/7/24.
//
#ifdef USE_OPENCV

#include <opencv2/opencv.hpp>
#include <matrix/include/utils/Logger.h>
using namespace cv;
using namespace matrix;
#endif
int main() {
#ifdef USE_OPENCV
    const std::string name = "/Users/jarlene/Desktop/image.png";

    Mat mat = cv::imread(name);

    Logger::Global()->Info("the image size is %d X %d channel is %d \n", mat.cols, mat.rows, mat.channels());
#endif
    return 0;
}

