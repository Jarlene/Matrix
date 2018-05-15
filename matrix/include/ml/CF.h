//
// Created by Jarlene on 2018/2/11.
//

#ifndef MATRIX_CF_H
#define MATRIX_CF_H


#include "BaseMl.h"

namespace matrix {

    template<class T = float>
    class CF : BaseMl<T> {
    public:
        void Train() override {

        }

        void Classify(const Mat<T>& test, Vec<T>& predictedLabels) override {

        }
    };

}

#endif //MATRIX_CF_H
