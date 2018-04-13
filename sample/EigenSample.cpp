//
// Created by Jarlene on 2017/7/25.
//

#include <iostream>
#include <matrix/include/base/Tensor.h>
#include <matrix/include/utils/Math.h>
#include <matrix/include/utils/Time.h>
#include <matrix/include/utils/Eigen.h>
using namespace matrix;
using namespace std;



int main() {
#ifdef USE_EIGEN
    //数组转矩阵
    float *aMat = new float[20];
    for (int i = 0; i < 20; i++) {
        aMat[i] = rand() % 11;
    }
    //静态矩阵，编译时确定维数 Matrix<double,4,5>

    Mat<float> staMat(aMat, 4, 5);
    //输出
    std::cout << staMat << std::endl;

    float a[] = {1, 2, 3,
                 4, 5, 6};
    float b[] = {2, 3,
                 4, 5,
                 6, 7};
    float *c = new float[4];


    Mat<float> am(a, 2, 3);
    Mat<float> bm(b, 3, 2);
    Mat<float> cm(c, 2, 2);

    cout << am << endl;
    cout << bm << endl;


    cm = am * bm;
    std::cout << cm << std::endl;
#endif
    return 0;

}
