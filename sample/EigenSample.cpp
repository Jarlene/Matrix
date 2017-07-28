//
// Created by Jarlene on 2017/7/25.
//

#include <eigen3/Eigen/Dense>
#include <iostream>
using Eigen::MatrixXd;
using namespace Eigen;
using namespace std;

int main() {

    //数组转矩阵
    float *aMat = new float[20];
    for (int i = 0; i < 20; i++) {
        aMat[i] = rand() % 11;
    }
    //静态矩阵，编译时确定维数 Matrix<double,4,5>
    Map<Matrix<float, 4, 5> > staMat(aMat);

    //输出
    std::cout << staMat << std::endl;

    float a[] = {1, 2, 3,
                 4, 5, 6};
    float b[] = {2, 3,
                 4, 5,
                 6, 7};
    float *c = new float[4];





    Map<Matrix<float, 2, 3, RowMajor>> am(a);
    Map<Matrix<float, 3, 2, RowMajor>> bm(b);
    Map<Matrix<float, 2, 2, RowMajor>> cm(c);

    cout << am << endl;
    cout << bm << endl;


    cm = am * bm;
    std::cout << cm << std::endl;
//
    return 0;

}