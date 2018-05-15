//
// Created by Jarlene on 2018/3/29.
//

#ifndef MATRIX_EIGEN_H
#define MATRIX_EIGEN_H
#ifndef USE_EIGEN
#pragma message("Fatal: Parameters servers need zeroMQ please set option use_zmp on")
#endif
#ifdef USE_EIGEN

#include "matrix/include/base/Tensor.h"
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
Mat<T> create(const matrix::Tensor<T> &tensor) {
    auto data = tensor.Data();
    auto shape = tensor.GetShape();
    assert(shape.isMatrix());
    return create<T>(data, shape[0], shape[1]);
}

/**
 * this will use eigen to manager memory
 * @tparam T
 * @param dim0
 * @param dim1
 * @return
 */
template<class T = float>
Mat<T> create(int dim0, int dim1) {
    typedef Matrix<T, Eigen::Dynamic, Eigen::Dynamic, RowMajor> Matrix;
    Matrix a(dim0, dim1);
    return Mat<T>(a.data(), dim0, dim1);
}

template<class T = float>
Mat<T> create(matrix::Shape &shape) {
    assert(shape.isMatrix());
    return create<T>(shape[0], shape[1]);
}

template<class T = float>
Mat<T> create(const matrix::Shape &shape) {
    assert(shape.isMatrix());
    return create<T>(shape[0], shape[1]);
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


template<class T = float>
Vec<T> create(const matrix::Tensor<T> &tensor) {
    auto data = tensor.Data();
    auto shape = tensor.GetShape();
    assert(shape.isVector());
    return create<T>(data, shape[0]);
}

/**
 * this will use eigen to manager memory
 * @tparam T
 * @param dim
 * @return
 */
template<class T = float>
Vec<T> create(int dim) {
    typedef Matrix<T, Eigen::Dynamic, 1> Vector;
    Vector v(dim);
    return Vec<T>(v.data(), dim);
}

template<class T = float>
Vec<T> create(matrix::Shape &shape) {
    assert(shape.isVector());
    return create<T>(shape.Size());
}

template<class T = float>
Vec<T> create(const matrix::Shape &shape) {
    assert(shape.isVector());
    return create<T>(shape.Size());
}

#endif
#endif //MATRIX_EIGEN_H
