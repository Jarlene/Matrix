//
// Created by Jarlene on 2017/7/18.
//

#ifndef MATRIX_TENSOR_H
#define MATRIX_TENSOR_H

#include "Shape.h"
#include "matrix/include/utils/Base.h"
namespace matrix {

    template <class T>
    class Tensor {
    public:
        Tensor() = default;

        Tensor(const T *ptr, const Shape &shape) : data_((T *) ptr), shape_(shape) {

        }

        Tensor(const Shape &shape) : data_(nullptr), shape_(shape) {

        }

        Tensor(const Tensor<T> &tensor) : shape_(tensor.shape_), data_(tensor.data_) {

        }

        const size_t Rank() const  {
            return this->shape_.Rank();
        }

        const size_t Size() const  {
            return this->shape_.Size();
        }

        const T *Data() const {
            return data_;
        }

        const T *Data(int offset) const {
            return data_+offset;
        }

        T* MutableData() {
            return data_;
        }

        T* MutableData(int offset) {
            return data_ + offset;
        }

        Tensor &operator=(const Tensor<T> other) {
            shape_ = other.shape_;
            data_ = other.data_;
            return *this;
        }

        bool operator==(T val) {
            for (int i = 0; i < shape_.Size(); ++i) {
                if (data_[i] != val) {
                    return false;
                }
            }
            return true;
        }


        const Shape &GetShape() const {
            return shape_;
        }

    private:
        T * data_ {nullptr};
        Shape shape_;

    };

    template <class T, typename... H>
    inline Tensor<T> TensorN(const T * data, H... args) {
        const int dims = sizeof...(args);
        const int len[dims] = {args...};
        Shape shape(len, dims);
        return Tensor<T>(data,shape);
    }

}


INSTANCE_CLASS(Tensor);


#endif //MATRIX_TENSOR_H
