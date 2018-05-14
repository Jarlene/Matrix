//
// Created by Jarlene on 2017/7/24.
//
#ifdef USE_OPENCV

#include <opencv2/opencv.hpp>
#include <matrix/include/utils/Logger.h>
#include <matrix/include/utils/FileUtil.h>
using namespace cv;
using namespace matrix;
using namespace std;
#endif

int main() {
#ifdef USE_OPENCV
    const std::string name = "/Users/jarlene/Desktop/ev.jpg";

    Mat mat = cv::imread(name);
    Mat mv[3];
    split(mat, mv);

    Mat result = mv[0] + mv[1] + mv[2];

    imwrite("/Users/jarlene/Desktop/out.jpg", InputArray(result));

    Logger::Global()->Info("the image size is %d X %d channel is %d \n", mat.cols, mat.rows, mat.channels());
#endif
    return 0;
}

