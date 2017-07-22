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
        Tensor(T *ptr, matrix::Shape<dimension> shape);
    };




}





#endif //MATRIX_TENSOR_H
