//
// Created by Jarlene on 2017/7/22.
//

#ifndef MATRIX_BLOB_H
#define MATRIX_BLOB_H

#include <cassert>

#include "matrix/include/base/Shape.h"
#include "matrix/include/base/Tensor.h"

namespace matrix {


    struct Blob {
        void* ptr_ = nullptr;
        std::function<void(void*)> destroy = nullptr;

        ~Blob() {
            if (destroy != nullptr) {
                destroy(ptr_);
            }
        }


        template <class T>
        Tensor<T> GeneratorTensor(const Shape shape_) const {
            return Tensor<T>(static_cast<T *>(ptr_), shape_);
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
