//
// Created by Jarlene on 2017/7/29.
//

#ifndef MATRIX_MOMENTOPTIMIZER_H
#define MATRIX_MOMENTOPTIMIZER_H

#include "BaseOptimizer.h"

namespace matrix {


    class MomentOptimizer : public BaseOptimizer {
    public:
        MomentOptimizer(float learning_rate = 0.01f, float mont = 0.9f);
        virtual std::vector<NodePtr> GeneratorUpdate(const std::map<NodePtr, NodePtr> &variableNodes) override ;
    private:

        float momentum;
    };
}

#endif //MATRIX_MOMENTOPTIMIZER_H
