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

    std::cout << staMat.row(0).sum() << std::endl;



    Mat<> aaa = create<>(2, 3);
    aaa.setRandom();
    cout << "aaa:\n  " << aaa << endl;



    float a[] = {1, 2, 3,
                 4, 5, 6};
    float b[] = {2, 3,
                 4, 5,
                 6, 7};
    float *c = new float[4];

    float d[] = {1, 2};


    Mat<float> am(a, 2, 3);
    Mat<float> bm(b, 3, 2);
    Mat<float> cm(c, 2, 2);
    Vec<float> dv(d, 2) ;

    cout << "am:\n  " << am << endl;
    cout << "bm:\n  "<< bm << endl;


    cm = am * bm;
    std::cout<<  "cm:\n  " << cm << std::endl;


    auto dm = am.colwise() + dv;
    cout<< "dm:\n  " << dm << endl;

    cout<< "em:\n  " << am.array() * am.array() << endl;

#endif
    return 0;

}
