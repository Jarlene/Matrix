//
// Created by Jarlene on 2017/7/18.
//

#ifndef MATRIX_TENSOR_H
#define MATRIX_TENSOR_H

#include "Shape.h"

namespace matrix {

    template <class T>
    class Tensor {
    public:
        Tensor(T *ptr, const Shape &shape);

        Tensor(const Tensor<T> &tensor);

        const int Rank() const ;

        const size_t Size() const ;


    private:
        T * data_;
        Shape shape_;

    };




}





#endif //MATRIX_TENSOR_H
