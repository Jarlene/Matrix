//
// Created by Jarlene on 2017/7/21.
//

#include "matrix/include/base/Shape.h"

namespace matrix {

    template <int dimension>
    void Shape<dimension>::reShape(const Shape<dimension> &s) {
#pragma unroll
        for (int i = 0; i < dimension; ++i) {
            this->shape_[i] = s.shape_[i];
        }
    }

    template <int dimension>
    const size_t Shape<dimension>::size() const {
        size_t total = 1;
#pragma unroll
        for (int i = 0; i < dimension; ++i) {
            total *= shape_[i];
        }
        return total;
    }

    template <int dimension>
    Shape<dimension>::Shape(const int *shape) {
#pragma unroll
        for (int i = 0; i < dimension; ++i) {
            this->shape_[i] = shape[i];
        }
    }

    template <int dimension>
    const int Shape<dimension>::operator[](int idx) const {
        if (dimension > idx) {
            return shape_[idx];
        }
        return 0;
    }

    template <int dimension>
    Shape<dimension>::Shape(const Shape<dimension> &shape) {
        this->shape_ = shape.shape_;
    }

}