//
// Created by Jarlene on 2018/2/8.
//

#ifndef MATRIX_BASEML_H
#define MATRIX_BASEML_H


#include "matrix/include/base/Tensor.h"

namespace matrix {

    template <class T>
    class BaseMl {
    public:
        virtual void Train() = 0;

        virtual void Classify(const Tensor<T>& test, Tensor<T>& predictedLabels) = 0;

    };

}
#endif //MATRIX_BASEML_H
