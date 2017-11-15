//
// Created by Jarlene on 2017/11/7.
//

#include "matrix/include/optimizer/SGDOptimizer.h"

namespace matrix {


    SGDOptimizer::SGDOptimizer(float learning_rate) : learning_rate(learning_rate){

    }

    std::vector<NodePtr> SGDOptimizer::GeneratorUpdate(const std::map<NodePtr, NodePtr> &variableNodes) {

        return std::vector<NodePtr>();
    }

}