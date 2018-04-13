//
// Created by Jarlene on 2018/3/29.
//

#ifndef MATRIX_EIGEN_H
#define MATRIX_EIGEN_H
#ifdef USE_EIGEN
#include <eigen3/Eigen/Eigen>
using namespace Eigen;
template <class T = float>
using Mat =  Map<Matrix<T, Dynamic, Dynamic, RowMajor>>;
template <class T = float>
using Vec =  Map<Matrix<T, Dynamic, 1>>;
#endif
#endif //MATRIX_EIGEN_H
