//
// Created by Jarlene on 2017/7/25.
//

#include <iostream>
#include <matrix/include/base/Tensor.h>
#include <matrix/include/utils/Math.h>
#ifdef USE_EIGEN
using Eigen::MatrixXd;
using namespace Eigen;
#endif
using namespace matrix;
using namespace std;

template <class T>
using Mat = Matrix<T, Dynamic, Dynamic, RowMajor>;

int main() {
#ifdef USE_EIGEN
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


    Mat<float> bma(4, 5);
    bma << 1, 2, 3, 4, 5, 6, 7, 7, 5, 4, 2, 1, 5, 6, 5, 3, 6, 3, 1, 1;
    cout << bma << endl;

    Map<Matrix<float, 2, 3, RowMajor>> am(a);
    Map<Matrix<float, 3, 2, RowMajor>> bm(b);
    Map<Matrix<float, 2, 2, RowMajor>> cm(c);

    cout << am << endl;
    cout << bm << endl;


    cm = am * bm;
    std::cout << cm << std::endl;

    auto A = TensorN(a,2,3);
    auto B = TensorN(b,3,2);
//    A.print();
#endif
    return 0;

}
