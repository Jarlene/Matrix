//
// Created by Jarlene on 2018/2/6.
//

#ifndef MATRIX_ADABOOST_H
#define MATRIX_ADABOOST_H

#include <iostream>
#include <vector>

#include "BaseMl.h"
#include "Perceptron.h"
#include "matrix/include/utils/MathTensor.h"
#include "matrix/include/op/ReduceOp.h"
#include "matrix/include/utils/Init.h"

namespace matrix {


    template<class WeakLearnerType = Perceptron<>, class T = float>
    class AdaBoost : public BaseMl<T>{
    public:
        AdaBoost(const Tensor<T> &data,
                 const Tensor<T> &labels,
                 const size_t numClasses,
                 const WeakLearnerType &wl,
                 const size_t iterations = 100,
                 const double tolerance = 1e-6) : data(data), labels(labels), numClasses(numClasses), wl(wl),
                                                  tolerance(tolerance), iterations(iterations) {

        }



        void Train() override {
            Train(data, labels, numClasses, wl, iterations, tolerance);
        }

        void Classify(const Tensor<T>& test, Tensor<T>& predictedLabels) override {

        }

    private:
        void Train(const Tensor<T> &data,
                   const Tensor<T> &labels,
                   const size_t numClasses,
                   const WeakLearnerType &learner,
                   const size_t iterations = 100,
                   const double tolerance = 1e-6) {

            this->learns.clear();
            this->weight.clear();
            this->tolerance = tolerance;
            this->numClasses = numClasses;

            Tensor<T> predictedLabels(labels.GetShape());
            InitTensor(predictedLabels);

            Tensor<T> tempData(data);

            Tensor<T> sumFinalH(ShapeN(numClasses, data.GetShape()[0]));
            InitTensor(sumFinalH);
            Value(sumFinalH, T(0));
            const double initWeight = 1.0 / (data.GetShape()[0] * numClasses);


            Tensor<T> D(ShapeN(numClasses, data.GetShape()[0]));
            InitTensor(D);
            Value(D, T(initWeight));


            Tensor<T> weights(labels.GetShape());
            InitTensor(weights);

            Tensor<T> finalH(labels.GetShape());
            InitTensor(finalH);

            double rt, crt = 0.0, alphat = 0.0, zt;
            for (int i = 0; i < iterations; ++i) {
                rt = 0.0;
                zt = 0.0;
                Sum(D, 0, weights);
                WeakLearnerType w(tempData, labels, numClasses, weights);
                w.Classify(tempData, predictedLabels);

                for (int j = 0; j < D.GetShape()[0]; ++j) {
                    if (predictedLabels.Data()[j] == labels.Data()[j]) {

                    } else {

                    }
                }
                
            }


            DestoryTensor(predictedLabels);
            DestoryTensor(sumFinalH);
            DestoryTensor(D);
            DestoryTensor(weights);
            DestoryTensor(finalH);
        }


    private:
        const Tensor<T> &data;
        const Tensor<T> &labels;
        const WeakLearnerType &wl;

        size_t numClasses;
        double tolerance;
        size_t iterations;

        std::vector<WeakLearnerType> learns;
        std::vector<double> weight;


    };


}

#endif //MATRIX_ADABOOST_H
