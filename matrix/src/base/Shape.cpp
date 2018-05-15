//
// Created by Jarlene on 2017/7/21.
//

#include "matrix/include/base/Shape.h"

namespace matrix {

    void Shape::reShape(const Shape &s) {
        if (s == *this) {
            return;
        }
        this->shape_.clear();
        this->shape_.insert(shape_.begin(), s.shape_.begin(), s.shape_.end());
    }

    const size_t Shape::Size() const {
        size_t total = 1;
#pragma unroll
        for (int i = 0; i < Rank(); ++i) {
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

    const int Shape::At(int idx) const {
        if (shape_.size() > idx) {
            return shape_[idx];
        }
        return 0;
    }

    Shape &Shape::operator=(const Shape &other) {
        if (*this == other) {
            return *this;
        }
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

    void Shape::Append(int idx) {
        shape_.push_back(idx);
    }

    const bool Shape::operator==(const Shape &shape) const {
        if (this->Size() != shape.Size()) {
            return false;
        }
        if (Rank() != shape.Rank()) {
            return false;
        }
        for (int i = 0; i < Rank(); ++i) {
            if (this->shape_[i] != shape.shape_[i]) {
                return false;
            }
        }
        return true;
    }

    const std::vector<int> &Shape::Array() const {
        return shape_;
    }

    const bool Shape::isConstant() const {
        return Rank() == 1 && Size() == 1;
    }

    const bool Shape::isVector() const {
        return Rank() == 1;
    }

    const bool Shape::isMatrix() const{
        return Rank() == 2;
    }

    const int Shape::StrideInclude(int idx) const {
        assert(idx >= 0);
        int result = 1;
#pragma unroll
        for (int i = idx; i < shape_.size(); ++i) {
            result *= shape_[i];
        }
        return result;
    }

    const int Shape::StrideExclude(int idx) const {
        assert(idx >= 0);
        int result = 1;
#pragma unroll
        for (int i = idx + 1; i < shape_.size(); ++i) {
            result *= shape_[i];
        }
        return result;
    }

}