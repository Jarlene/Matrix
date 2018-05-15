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

        SVD(const Mat<T> & dataset,
            Mat<T>& u,
            Mat<T>& v,
            Mat<T>& sigma,
            const double epsilon = 0.03,
            const double delta = 0.1) : dataset(&dataset) {

        }

        void ExtractSVD(Mat<T>& u, Mat<T>& v, Mat<T>& sigma) {

        }

        void Train() override {

        }

        void Classify(const Mat<T>& test, Vec<T>& predictedLabels) override {

        }

    private:
        const Mat<T> *dataset;
        Mat<T> basis;
    };

}

#endif //MATRIX_SVD_H
