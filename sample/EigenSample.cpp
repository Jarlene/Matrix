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

    const bool  aa = false;
    
    //动态矩阵，运行时确定 MatrixXd
    int a = aa?4:5;
    int b = aa?5:4;
//    Map<MatrixXd> dymMat(aMat, a, b);


//    std::cout<< staMat << std::endl;
//    std::cout<< dymMat << std::endl;

//    auto  s = staMat * dymMat.transpose();

    //输出，应该和上面一致

//    std::cout<< s << std::endl;

    return 0;

}