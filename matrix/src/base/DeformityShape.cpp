//
// Created by Jarlene on 2017/12/14.
//

#include "matrix/include/base/DeformityShape.h"


namespace matrix {

    void DeformityShape::Append(int idx, int val) {
        shape_.at(idx).push_back(val);
    }

    void DeformityShape::Append(int val) {
        Append(0, val);
    }

    const size_t DeformityShape::Rank() const {
        return shape_.size();
    }

    const size_t DeformityShape::Size() const {
        size_t  t = 1;
        for(auto it : shape_) {
           for (auto subit : it) {
               t *= subit;
           }
        }
        return t;
    }

    const Shape DeformityShape::At(int level) const {
        if (Rank() <= level) {
            return Shape();
        }
        return Shape(shape_[level].data(), shape_[level].size());
    }

    void DeformityShape::ReShape(const std::vector<Shape *> &shapes) {
        for (auto s : shapes) {
            shape_.push_back(s->Array());
        }
    }

    void DeformityShape::ReShape(const std::vector<Shape> &shapes) {
        for (auto s : shapes) {
            shape_.push_back(s.Array());
        }
    }

    const int DeformityShape::At(int idx, int val) const {
        if(idx > shape_.size() || idx < 0) {
            return -1;
        }
        int size = shape_.at(idx).size();
        if (val > size || val < 0) {
            return -1;
        }
        return shape_.at(idx).at(val);
    }

    DeformityShape &DeformityShape::operator=(const DeformityShape &other) {
        if (other == *this) {
            return *this;
        }
        this->shape_.clear();
#pragma unroll
        for (auto i: other.shape_) {
            shape_.push_back(i);
        }
        return *this;
    }

    DeformityShape &DeformityShape::operator=(const Shape &other) {
        if (other == *this) {
            return *this;
        }
        this->shape_.clear();
        shape_.push_back(other.Array());
        return *this;
    }

    const bool DeformityShape::operator==(const Shape &shape) const {
        if (this->Rank() != 1) {
            return false;
        }
        if (this->Size() != shape.Size()) {
            return false;
        }

        for (int i = 0; i < shape.Rank(); ++i) {
            if (shape[i] != shape_[0][i]) {
                return false;
            }
        }
        return true;
    }

    const bool DeformityShape::operator==(const DeformityShape &shape) const {
        if (this->Rank() != shape.Rank()) {
            return false;
        }
        if (this->Size() != shape.Size()) {
            return false;
        }

        for (int i = 0; i < Rank(); ++i) {
            if (!(this->At(i) == shape[i])) {
                return false;
            }
        }
        return true;
    }

    const Shape DeformityShape::operator[](int level) const {
        if (level >= Rank()) {
            return Shape();
        }
        return Shape(shape_[level].data(), shape_[level].size());
    }


}