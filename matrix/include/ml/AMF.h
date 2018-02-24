//
// Created by Jarlene on 2018/2/8.
//

#ifndef MATRIX_AMF_H
#define MATRIX_AMF_H

#include "BaseMl.h"

namespace matrix {

    template <class T = float>
    class AMF : public BaseMl<T> {
    public:
        AMF() {

        }

        void Train() override {

        }

        void Classify(const Tensor<T> &test, Tensor<T> &predictedLabels) override {

        }
    };


}

#endif //MATRIX_AMF_H
