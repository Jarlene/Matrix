//
// Created by Jarlene on 2018/2/6.
//

#ifndef MATRIX_PERCEPTRON_H
#define MATRIX_PERCEPTRON_H


#include "matrix/include/base/Tensor.h"

namespace matrix {

    class LearnPolicy {
    public:
        template <class T>
        void UpdateWeights(const T& trainingPoint,
                           Tensor<T>& weights,
                           Tensor<T>& biases,
                           const size_t incorrectClass,
                           const size_t correctClass,
                           const double instanceWeight = 1.0) {

        }
    };


    template <class policy = LearnPolicy,  class T = float>
    class Perceptron : BaseMl<T> {
    public:
        Perceptron(const Mat<T> &data,
                   const Vec<T> &labels,
                   const size_t numClasses,
                   const Mat<T> &instanceWeights) : data(data), labels(labels),
                                                                     numClasses(numClasses),
                                                                     instanceWeights(instanceWeights) {

        }


        void Train() {

        }

        void Classify(const Mat<T>& test, Vec<T>& predictedLabels) {

        }


    private:
        const Mat<T>& data;
        const Vec <T>& labels;
        const size_t numClasses;
        const Mat<T>& instanceWeights;
    };




}

#endif //MATRIX_PERCEPTRON_H
