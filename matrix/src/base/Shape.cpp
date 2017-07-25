//
// Created by Jarlene on 2017/7/21.
//

#include "matrix/include/base/Shape.h"

namespace matrix {

    void Shape::reShape(const Shape &s) {
#pragma unroll
        for (int i = 0; i < s.size(); ++i) {
            this->shape_[i] = s.shape_[i];
        }
    }

    const size_t Shape::size() const {
        size_t total = 1;
#pragma unroll
        for (int i = 0; i < shape_.size(); ++i) {
            total *= shape_[i];
        }
        return total;
    }

    Shape::Shape(const int *shape) {
#pragma unroll
        for (int i = 0; i < shape_.size(); ++i) {
            this->shape_[i] = shape[i];
        }
    }

    const int Shape::operator[](int idx) const {
        if (shape_.size() > idx) {
            return shape_[idx];
        }
        return 0;
    }

    Shape::Shape(const Shape &shape) {
        this->shape_ = shape.shape_;
    }

}