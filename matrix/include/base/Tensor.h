//
// Created by Jarlene on 2017/7/18.
//

#ifndef MATRIX_TENSOR_H
#define MATRIX_TENSOR_H

#include "Shape.h"

namespace matrix {

    template <class T, int dimension>
    class Tensor {
    public:
        Tensor(T *ptr, Shape <dimension> shape);

        Tensor(const Tensor<T, dimension> &tensor);

        const int Rank() const ;

        const size_t Size() const ;


    private:
        T * data_;
        Shape<dimension> shape_;

    };




}





#endif //MATRIX_TENSOR_H
