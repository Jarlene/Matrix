//
// Created by Jarlene on 2017/12/14.
//

#include "matrix/include/base/LoShape.h"


namespace matrix {

    void LoShape::Append(int idx, int val) {
        shape_.at(idx).push_back(val);
    }

    void LoShape::Append(int val) {
        Append(0, val);
    }

    const size_t LoShape::Rank() const {
        return shape_.size();
    }

    const size_t LoShape::Size() const {
        size_t  t = 1;
        for(auto it : shape_) {
           for (auto subit : it) {
               t *= subit;
           }
        }
        return t;
    }

    const Shape LoShape::At(int level) const {
        assert(Rank() > level);
        return Shape(shape_[level].data(), shape_[level].size());
    }

    void LoShape::ReShape(const std::vector<Shape *> &shapes) {
        for (auto s : shapes) {
            shape_.push_back(s->Array());
        }
    }

    void LoShape::ReShape(const std::vector<Shape> &shapes) {
        for (auto s : shapes) {
            shape_.push_back(s.Array());
        }
    }

    const int LoShape::At(int idx, int val) const {
        if(idx > shape_.size() || idx < 0) {
            return -1;
        }
        int size = shape_.at(idx).size();
        if (val > size || val < 0) {
            return -1;
        }
        return shape_.at(idx).at(val);
    }

    LoShape &LoShape::operator=(const LoShape &other) {
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

    LoShape &LoShape::operator=(const Shape &other) {
        if (other == *this) {
            return *this;
        }
        this->shape_.clear();
        shape_.push_back(other.Array());
        return *this;
    }

    const bool LoShape::operator==(const Shape &shape) const {
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

    const bool LoShape::operator==(const LoShape &shape) const {
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

    const Shape LoShape::operator[](int level) const {
        assert(Rank() > level);
        return Shape(shape_[level].data(), shape_[level].size());
    }


}