//
// Created by Jarlene on 2017/7/29.
//

#ifndef MATRIX_ADMAXOPTIMIZER_H
#define MATRIX_ADMAXOPTIMIZER_H

#include "BaseOptimizer.h"

namespace matrix {

    class AdmaxOptimizer : public BaseOptimizer {
    public:
        AdmaxOptimizer(float learning_rate);

        virtual std::vector<NodePtr> GeneratorUpdate(const std::map<NodePtr, NodePtr> &variableNodes) override ;
    };
}

#endif //MATRIX_ADMAXOPTIMIZER_H
