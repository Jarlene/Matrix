//
// Created by Jarlene on 2018/3/29.
//

#ifndef MATRIX_EIGEN_H
#define MATRIX_EIGEN_H
#ifdef USE_EIGEN

#include <eigen3/Eigen/Eigen>

using namespace Eigen;
template<class T = float>
using Mat =  Map<Matrix<T, Dynamic, Dynamic, RowMajor>>;
template<class T = float>
using Vec =  Map<Matrix<T, Dynamic, 1>>;


template<class T = float>
Mat<T> create(const T *data, int dim0, int dim1) {
    T *a = const_cast<T *>(data);
    return Mat<T>(a, dim0, dim1);
}

template<class T = float>
Mat<T> create(T *data, int dim0, int dim1) {
    return Mat<T>(data, dim0, dim1);
}


template<class T = float>
Vec<T> create(const T *data, int dim) {
    T *a = const_cast<T *>(data);
    return Vec<T>(a, dim);
}

template<class T = float>
Vec<T> create(T *data, int dim) {
    return Vec<T>(data, dim);
}


#endif
#endif //MATRIX_EIGEN_H