//
// Created by Jarlene on 2017/7/21.
//

#include "matrix/include/base/Tensor.h"



namespace matrix {

    template <class T>
    Tensor<T>::Tensor(T *ptr, const Shape &shape) :shape_(shape), data_(ptr) {

    }

    template <class T>
    Tensor<T>::Tensor(const Tensor<T> &tensor) :shape_(tensor.shape_), data_(tensor.data_) {

    }

    template <class T>
    const int Tensor<T>::Rank() const {
        return shape_.size();
    }

    template <class T>
    const size_t Tensor<T>::Size() const {
        return this->shape_.size();
    }



}
