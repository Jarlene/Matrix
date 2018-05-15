//
// Created by Jarlene on 2018/5/15.
//

#ifndef MATRIX_PCA_H
#define MATRIX_PCA_H

#include "BaseMl.h"

namespace matrix {
    template <class T = float>
    class PCA : public BaseMl<T> {
    public:
        void Train() override {

        }

        void Classify(const Mat<T> &test, Vec<T> &predictedLabels) override {

        }
    };
}

#endif //MATRIX_PCA_H
