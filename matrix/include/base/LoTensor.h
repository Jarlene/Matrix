//
// Created by Jarlene on 2017/12/18.
//

#ifndef MATRIX_LOTENSOR_H
#define MATRIX_LOTENSOR_H

#include "DeformityShape.h"
#include "Tensor.h"
#include "matrix/include/utils/Base.h"

namespace matrix {

    template <class T>
    class LoTensor : public Tensor<T> {
    public:
        LoTensor() = default;
    };

}

#endif //MATRIX_LOTENSOR_H
