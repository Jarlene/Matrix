//
// Created by Jarlene on 2017/7/29.
//

#ifndef MATRIX_BASEOPTIMIZER_H
#define MATRIX_BASEOPTIMIZER_H

#include <map>
#include "matrix/include/base/Node.h"
#include "matrix/include/utils/Math.h"

namespace matrix {

    class BaseOptimizer {
    public:
        BaseOptimizer(float learning_rate) : learning_rate(learning_rate) {

        };

        BaseOptimizer(const BaseOptimizer &) = default;

        BaseOptimizer(BaseOptimizer &&) = default;

        BaseOptimizer &operator=(const BaseOptimizer &) = default;

        BaseOptimizer &operator=(BaseOptimizer &&) = default;

        virtual ~BaseOptimizer() = default;

        virtual std::vector<NodePtr> GeneratorUpdate(const std::map<NodePtr, NodePtr> &variableNodes) = 0;

        void SetDecay(float decay) {
            this->decay = decay;
        }

    protected:
        float decay = 0.0f;
        float learning_rate;
    };

}


#endif //MATRIX_BASEOPTIMIZER_H
