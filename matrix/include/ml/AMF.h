//
// Created by Jarlene on 2018/2/8.
//

#ifndef MATRIX_AMF_H
#define MATRIX_AMF_H

#include "BaseMl.h"
#include "matrix/include/utils/Logger.h"

namespace matrix {

    template <class T = float>
    class AMF : public BaseMl<T> {
    public:
        AMF() {

        }

        void Train() override {
            LOG(FATAL) << "not implement";
        }

        void Classify(const Mat<T>& test, Vec<T>& predictedLabels) override {
            LOG(FATAL) << "not implement";
        }

        void Apply(const Mat<T>& data, const size_t rank, Mat<T>& W, Mat<T>& H) {
            const long n = data.rows();
            const long m = data.cols();
            W.Random();
            H.Random();
        }
    };


}

#endif //MATRIX_AMF_H
