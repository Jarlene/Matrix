//
// Created by Jarlene on 2017/7/25.
//

#include <eigen3/Eigen/Dense>
#include <iostream>
using Eigen::MatrixXd;
using namespace Eigen;
using namespace std;

int main() {
    MatrixXd m(2,2);
    m(0,0) = 3;
    m(1,0) = 2.5;
    m(0,1) = -1;
    m(1,1) = m(1,0) + m(0,1);

    cout << m << endl;



    m = MatrixXd::Random(3,3);
    m = (m + MatrixXd::Constant(3,3,1.2));
    cout << "m =" << endl << m << endl;
    VectorXd v(3);
    v << 1, 2, 3;

    cout << v <<endl;

    cout << "m * v =" << endl << m * v << endl;

    return 0;

}