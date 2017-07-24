//
// Created by 郑珊 on 2017/7/22.
//

#ifndef MATRIX_BLOB_H
#define MATRIX_BLOB_H

#include <cassert>

#include "matrix/include/base/Shape.h"
#include "matrix/include/base/Tensor.h"

namespace matrix {


    struct Blob {
        void* ptr_ = nullptr;
        Shape shape_;

        std::function<void(void*)> destory = nullptr;

        ~Blob() {
            if (destory != nullptr) {
                destory(ptr_);
            }
        }


        template <class T, int dimension>
        Tensor<T,dimension> Generator() const {
            return Tensor<T, dimension>(static_cast<T *>(ptr_), shape_);
        }

        template <class T, int dimension>
        Tensor<T, dimension> Generator(const Shape &other) const {
            assert(this->shape_.size() == other.size());
            return Tensor<T, dimension>(static_cast<T *>(ptr_), other);
        }

        template <class T>
        const T &Get() const {
            return *static_cast<T*>(ptr_);
        }

        template <class T>
        T * GetMutable() {
            return static_cast<T*>(ptr_);
        }

    };



}

#endif //MATRIX_BLOB_H
