//
// Created by Jarlene on 2018/2/6.
//

#ifndef MATRIX_GBDT_H
#define MATRIX_GBDT_H

#include "BaseMl.h"

namespace matrix {


    template <class T = float>
    class GBDT : public BaseMl<T>{
    public:
        void Train() override {

        }

        void Classify(const Mat<T>& test, Vec<T>& predictedLabels) override {

        }
    };
}

#endif //MATRIX_GBDT_H
