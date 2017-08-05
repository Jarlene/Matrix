//
// Created by Jarlene on 2017/7/21.
//

#include "matrix/include/base/Shape.h"

namespace matrix {

    void Shape::reShape(const Shape &s) {
#pragma unroll
        for (int i = 0; i < s.Size(); ++i) {
            this->shape_[i] = s.shape_[i];
        }
    }

    const size_t Shape::Size() const {
        size_t total = 1;
#pragma unroll
        for (int i = 0; i < shape_.size(); ++i) {
            total *= shape_[i];
        }
        return total;
    }

    const size_t Shape::Rank() const  {
        return shape_.size();
    }


    Shape::Shape(const int *shape, const int dim) {
#pragma unroll
        for (int i = 0; i < dim; ++i) {
            this->shape_.push_back(shape[i]);
        }
    }

    const int Shape::operator[](int idx) const {
        if (shape_.size() > idx) {
            return shape_[idx];
        }
        return 0;
    }

    Shape &Shape::operator=(const Shape &other) {
        this->shape_.clear();
#pragma unroll
        for (int i: other.shape_) {
            shape_.push_back(i);
        }
        return *this;
    }

    Shape::Shape(const Shape &shape) {
        this->shape_ = shape.shape_;
    }

    const bool Shape::operator==(const Shape &shape) const {
        if (this->Size() != shape.Size()) {
            return false;
        }
        for (int i = 0; i < Rank(); ++i) {
            if (this->shape_[i] != shape.shape_[i]) {
                return false;
            }
        }
        return true;
    }

}