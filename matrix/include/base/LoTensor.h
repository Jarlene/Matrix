//
// Created by Jarlene on 2017/12/18.
//

#ifndef MATRIX_LOTENSOR_H
#define MATRIX_LOTENSOR_H

#include "LoShape.h"
#include "Tensor.h"
#include "matrix/include/utils/Base.h"

namespace matrix {

    template <class T>
    class LoTensor : public Tensor<T> {
    public:
        LoTensor() = default;
        LoTensor(const T *ptr, const LoShape &shape) : data_(ptr), shape(shape) {

        }

        LoTensor( const LoShape &shape) : data_(nullptr), shape(shape) {

        }

        LoTensor(const LoTensor<T> &tensor) : shape(tensor.shape), data_(tensor.data_) {

        }

        LoTensor(const Tensor<T> &tensor) : shape(tensor.GetShape()), data_(tensor.Data()) {

        }

    private:
        LoShape shape;
        T * data_ {nullptr};
    };

}

#endif //MATRIX_LOTENSOR_H
