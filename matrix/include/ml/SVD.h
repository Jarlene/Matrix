//
// Created by Jarlene on 2018/2/8.
//

#ifndef MATRIX_SVD_H
#define MATRIX_SVD_H

#include "BaseMl.h"

namespace matrix {


    template <class T = float>
    class SVD  : public BaseMl<T>{
    public:

        SVD(const Tensor<T> & dataset,
            Tensor<T>& u,
            Tensor<T>& v,
            Tensor<T>& sigma,
            const double epsilon = 0.03,
            const double delta = 0.1) : dataset(&dataset) {

        }

        void ExtractSVD(Tensor<T>& u, Tensor<T>& v, Tensor<T>& sigma) {

        }

        void Train() override {

        }

        void Classify(const Tensor<T> &test, Tensor<T> &predictedLabels) override {

        }

    private:
        const Tensor<T> *dataset;
        Tensor<T> basis;
    };

}

#endif //MATRIX_SVD_H
