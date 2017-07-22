//
// Created by Jarlene on 2017/7/21.
//

#include "matrix/include/base/Shape.h"

namespace matrix {

    template <int dimension>
    void Shape<dimension>::reShape(const Shape &shape) {

    }

    template <int dimension>
    const size_t Shape<dimension>::size() const {
        return 0;
    }

    template <int dimension>
    Shape<dimension>::Shape(const int *shape) {
#pragma unroll
        for (int i = 0; i < dimension; ++i) {
            this->shape_[i] = shape[i];
        }
    }


}

template class matrix::Shape<4>;