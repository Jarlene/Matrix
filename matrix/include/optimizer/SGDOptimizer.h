//
// Created by Jarlene on 2017/7/29.
//

#ifndef MATRIX_SGDOPTIMIZER_H
#define MATRIX_SGDOPTIMIZER_H

#include "BaseOptimizer.h"

namespace matrix {

    class SGDOptimizer : public BaseOptimizer<0> {
    public:
        explicit SGDOptimizer(float learning_rate = 0.001);
        virtual void Update(std::vector<Blob> dx, std::vector<Blob> x) override ;
    private:
        float learning_rate;
    };
}

#endif //MATRIX_SGDOPTIMIZER_H
