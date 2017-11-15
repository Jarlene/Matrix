//
// Created by Jarlene on 2017/7/29.
//

#ifndef MATRIX_SGDOPTIMIZER_H
#define MATRIX_SGDOPTIMIZER_H

#include "BaseOptimizer.h"

namespace matrix {

    class SGDOptimizer : public BaseOptimizer {
    public:
        explicit SGDOptimizer(float learning_rate = 0.001);
        virtual std::vector<NodePtr> GeneratorUpdate(const std::map<NodePtr, NodePtr> &variableNodes) override ;
    private:
        float learning_rate;
    };
}

#endif //MATRIX_SGDOPTIMIZER_H
