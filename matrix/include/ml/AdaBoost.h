//
// Created by Jarlene on 2018/2/6.
//

#ifndef MATRIX_ADABOOST_H
#define MATRIX_ADABOOST_H

#include <iostream>
#include <vector>
#include <matrix/include/store/MemoryManager.h>
#include "matrix/include/base/Tensor.h"

namespace matrix {


    template<class WeakLearnerType, class T>
    class AdaBoost {
    public:
        AdaBoost(const Tensor<T> &data,
                 const Tensor<T> &labels,
                 const size_t numClasses,
                 const WeakLearnerType &wl,
                 const size_t iterations = 100,
                 const double tolerance = 1e-6) {

            this->predictedLabels = MemoryManager::Global()->GetCpuMemoryPool()->dynamicAllocate(
                    sizeof(T) * labels.Size());

            Train(data, labels, numClasses, wl, iterations, tolerance);
        }

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

            Tensor<T> predictedLabels(this->predictedLabels, labels.GetShape());

            const double initWeight = 1.0 / (data.GetShape()[0] * numClasses);


        }

    private:
        size_t numClasses;
        double tolerance;
        std::vector<WeakLearnerType> learns;
        std::vector<double> weight;

        T *predictedLabels;

    };


}

#endif //MATRIX_ADABOOST_H
