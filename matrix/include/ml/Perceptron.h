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


    template <class policy = LearnPolicy, class WeightInitializationPolicy,class T>
    class Perceptron {
    public:

        void Train(const Tensor<T>& data,
                   const Tensor<T>& labels,
                   const size_t numClasses,
                   const Tensor<T>& instanceWeights = Tensor<T>()) {

        }

        void Classify(const Tensor<T>& test, Tensor<T>& predictedLabels) {

        }
    };




}

#endif //MATRIX_PERCEPTRON_H
