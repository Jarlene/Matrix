//
// Created by Jarlene on 2018/2/6.
//

#ifndef MATRIX_ADABOOST_H
#define MATRIX_ADABOOST_H

#include <iostream>
#include <vector>

#include "BaseMl.h"
#include "Perceptron.h"
namespace matrix {


    template<class WeakLearnerType = Perceptron<>, class T = float>
    class AdaBoost : public BaseMl<T>{
    public:
        AdaBoost(const Mat<T> &data,
                 const Vec<T> &labels,
                 const size_t numClasses,
                 const size_t iterations = 100,
                 const double tolerance = 1e-6) : data(data), labels(labels), numClasses(numClasses),
                                                  tolerance(tolerance), iterations(iterations) {

        }



        void Train() override {
            Train(data, labels, numClasses, iterations, tolerance);
        }

        void Classify(const Mat<T>& test, Vec<T>& predictedLabels) override {

        }

    private:
        void Train(const Mat<T> &data,
                   const Vec<T> &labels,
                   const size_t numClasses,
                   const size_t iterations = 100,
                   const double tolerance = 1e-6) {

            this->learns.clear();
            this->weight.clear();
            this->tolerance = tolerance;
            this->numClasses = numClasses;

            // predict labels
            Vec<T> predictedLabels = create<T>(labels.rows());
            Mat<T> tempData = data;
            Mat<T> sumFinalH = create<T>(numClasses, predictedLabels.rows());
            sumFinalH.Zero();

            const T initWeight = T(1.0f / (data.rows() * numClasses));
            Mat<T> D = create<T>(numClasses, data.rows());
            D.fill(initWeight);


            Mat<T> weights = create<T>(labels.rows(), 1);
            Vec<T> finalH = create<T>(labels.rows());

            double rt, crt = 0.0, alphat = 0.0, zt;
            for (int i = 0; i < iterations; ++i) {
                rt = 0.0;
                zt = 0.0;
                weights.fill(D.sum());
                WeakLearnerType w(tempData, labels, numClasses, weights);
                w.Classify(tempData, predictedLabels);

                for (int j = 0; j < D.cols(); ++j) {
                    if (predictedLabels[j] == labels[j]) {
                        rt += D.col(j).sum();
                    } else {
                        rt -= D.col(j).sum();
                    }
                }
                if ((i > 0) && (std::abs(rt - crt) < tolerance))
                    break;

                if (rt >= 1.0) {
                    weight.push_back(1.0);
                    learns.push_back(w);
                    break;
                }

                crt = rt;
                alphat = 0.5 * log((1 + rt) / (1 - rt));
                weight.push_back(alphat);
                learns.push_back(w);

                for (int j = 0; j < D.cols(); ++j) {
                    const double expo = exp(alphat);
                    if (predictedLabels[j] == labels[j]) {
                        for (int k = 0; k < D.rows(); ++k) {
                            D(k, j) /= expo;
                            zt += D(k, j);
                            if (k == static_cast<int>(labels[j]))
                                sumFinalH(k, j) += (alphat);
                            else
                                sumFinalH(k, j) -= (alphat);
                        }
                    } else {
                        for (int k = 0; k < D.rows(); ++k) {
                            D(k, j) *= expo;
                            zt += D(k, j);
                            if (k == static_cast<int>(labels[j]))
                                sumFinalH(k, j) += (alphat);
                            else
                                sumFinalH(k, j) -= (alphat);
                        }
                    }
                }

                D /= zt;
            }

        }


    private:
        const Mat<T> &data;
        const Vec<T> &labels;
        size_t numClasses;
        double tolerance;
        size_t iterations;

        std::vector<WeakLearnerType> learns;
        std::vector<double> weight;


    };


}

#endif //MATRIX_ADABOOST_H
