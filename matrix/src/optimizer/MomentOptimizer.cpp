//
// Created by Jarlene on 2017/11/7.
//

#include "matrix/include/optimizer/MomentOptimizer.h"

namespace matrix {

    MomentOptimizer::MomentOptimizer(float learning_rate, float mont) : BaseOptimizer(learning_rate), momentum(mont) {

    }

    std::vector<NodePtr> MomentOptimizer::GeneratorUpdate(const std::map<NodePtr, NodePtr> &variableNodes) {
        std::vector<NodePtr> result;
        for(auto &it : variableNodes) {
            auto node = Node::Create();
            node->opName = "applyGrad";
            node->params["learning_rate"] = learning_rate;
            node->params["momentum"] = momentum;
            node->params["type"] = kMomentum;
            node->params["decay"] = decay;
            node->inputs.push_back(it.first);
            node->inputs.push_back(it.second);
            it.first->outputs.push_back(std::weak_ptr<Node>(node));
            it.second->outputs.push_back(std::weak_ptr<Node>(node));
            node->nodeName = it.first->nodeName + "_apply_" + it.second->nodeName;
            node->Build();
            result.push_back(node);
        }
        return result;
    }

}