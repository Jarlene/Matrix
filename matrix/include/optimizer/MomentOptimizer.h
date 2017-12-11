//
// Created by Jarlene on 2017/7/29.
//

#ifndef MATRIX_MOMENTOPTIMIZER_H
#define MATRIX_MOMENTOPTIMIZER_H

#include "BaseOptimizer.h"

namespace matrix {


    class MomentOptimizer : public BaseOptimizer {
    public:
        MomentOptimizer(float learning_rate = 0.0001f, float mont = 0.005f);
        virtual std::vector<NodePtr> GeneratorUpdate(const std::map<NodePtr, NodePtr> &variableNodes) override ;
    private:
        float learning_rate;
        float momentum;
    };
}

#endif //MATRIX_MOMENTOPTIMIZER_H
