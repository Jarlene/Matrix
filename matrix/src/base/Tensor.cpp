//
// Created by Jarlene on 2017/7/21.
//

#include "matrix/include/base/Tensor.h"



namespace matrix {

    template <class T, int dimension>
    Tensor<T, dimension>::Tensor(T *ptr, Shape <dimension> shape) :shape_(shape), data_(ptr) {

    }

    template <class T, int dimension>
    Tensor<T, dimension>::Tensor(const Tensor<T, dimension> &tensor) :shape_(tensor.shape_), data_(tensor.data_) {

    }

    template <class T, int dimension>
    const int Tensor<T, dimension>::Rank() const {
        return dimension;
    }

    template <class T, int dimension>
    const size_t Tensor<T, dimension>::Size() const {
        return this->shape_.size();
    }



}

template class matrix::Tensor<double, 2>;
