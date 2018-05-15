//
// Created by Jarlene on 2018/5/15.
//

#ifndef MATRIX_HMM_H
#define MATRIX_HMM_H

#include "BaseMl.h"

namespace matrix {

    template <class T = float>
    class HMM : public BaseMl<T> {
    public:
        void Train() override {

        }

        void Classify(const Mat<T> &test, Vec<T> &predictedLabels) override {

        }
    };

}

#endif //MATRIX_HMM_H
