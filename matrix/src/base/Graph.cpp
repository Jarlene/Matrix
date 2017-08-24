//
// Created by Jarlene on 2017/8/9.
//

#include "matrix/include/executor/GraphAlgorithm.h"
#include "matrix/include/base/Graph.h"
#include "matrix/include/store/MemoryManager.h"

namespace matrix {


    void static GraphAddNodes(std::vector<NodePtr> &vector, const NodePtr &node) {

        auto it = std::find(vector.begin(), vector.end(), node);
        if (it == vector.end()) {
            vector.push_back(node);
        }

        for (auto &e : node->inputs) {
            GraphAddNodes(vector, e);
        }

    }

    Graph::Graph(const Symbol &symbol, bool isTrain) {
        if (isTrain) {
            GeneratorGradNodes(symbol);
        }
        GraphAddNodes(nodes_, symbol.GetNode());
    }

    Graph::~Graph() {
        MemoryManager::Global()->GetCpuMemoryPool()->freeAll();
    }

    NodePtr Graph::GetNode(const std::string &name) {
        for (auto &item : nodes_) {
            if (item->opName == name) {
                return item;
            }
        }
        return nullptr;
    }

    NodePtr Graph::GetNode(size_t id) {
        for (auto &item : nodes_) {
            if (item->id_ == id) {
                return item;
            }
        }
        return nullptr;
    }

    void Graph::Optimize() {
        Unique();
    }

    void Graph::AllocateGraph(const std::vector<NodePtr> &fetch) {
        for(auto node : nodes_) {
            if (node->isVariable && node->isBackward) {
                node->data_ = MemoryManager::Global()->GetCpuMemoryPool()->dynamicAllocate(node->memorySize);
            }
            if (node->op != nullptr && node->memorySize > 0 && node->data_ == nullptr) {
                node->data_ = MemoryManager::Global()->GetCpuMemoryPool()->dynamicAllocate(node->memorySize);
            }
        }
//        GraphAlgorithm colorGraph;
//        colorGraph.Coloring(*this, graphColor_, fetch);
//
//        std::map<int, size_t> colorSize;
//
//        std::unordered_map<int, std::vector<int>> colors;
//
//        for (auto &it : graphColor_) {
//            int color = it.second;
//            colors[color].push_back(it.first);
//        }
//
//        for (auto &it : colors) {
//            auto vec = it.second;
//            long maxSize = 0;
//            for (int id : vec) {
//                auto node = GetNode(id);
//                if (maxSize < node->getMemorySize()) {
//                    maxSize = node->getMemorySize();
//                }
//            }
//            colorSize[it.first] = maxSize;
//        }
//        MemoryManager::Global()->GetCpuMemoryPool()->staticAllocate(colorSize);
//        for (auto &it : graphColor_) {
//            auto node = GetNode(it.first);
//            int color = it.second;
//            if (node->data_ == nullptr) {
//                node->data_= MemoryManager::Global()->GetCpuMemoryPool()->getMemory(color);
//            }
//        }
//        for (auto &item : nodes_) {
//
//
//            if (item->data_ == nullptr && item->op != nullptr) {
//                size_t m = item->getMemorySize();
//                if (m > 0) {
//                    item->data_ = MemoryManager::Global()->GetCpuMemoryPool()->dynamicAllocate(m);
//                } else {
//                    Logger::Global()->Info("%s not in memory \n", item->nodeName.c_str());
//                }
//
//            }
//        }
//        MemoryManager::Global()->GetCpuMemoryPool()->PrintMemory();
    }

    void Graph::SaveVariableData(std::string &file) {

    }

    void Graph::SaveModel(std::string &file) {

    }

    void Graph::AppendNode(const NodePtr &node) {

    }

    const std::vector<NodePtr> &Graph::GetGraphNodes() const {
        return nodes_;
    }

    void Graph::Unique() {
        sort(nodes_.begin(), nodes_.end(), less);
        nodes_.erase(unique(nodes_.begin(), nodes_.end()), nodes_.end());
    }

    void Graph::GeneratorGradNodes(const Symbol &symbol) {
        std::map<NodePtr, NodePtr> gradMap;
        auto out = symbol.GetNode();
        auto ones = Node::Create();
        ones->opName = "variable";
        ones->nodeName = "ones";
        ones->outputShapes = out->outputShapes;
        ones->context = out->context;
        ones->params["constant"] = 1;
        ones->Build();
        gradMap[out] = ones;

        std::vector<NodePtr> stack;
        stack.push_back(out);

        while(!stack.empty()) {
            auto pre = stack.back();
            stack.pop_back();
            int index = 0;
            for (auto &item : pre->inputs) {
                auto grad_node = item->GetGradNode(index, pre, gradMap[pre]);
                gradMap[item] = grad_node;
                auto it = std::find(stack.begin(), stack.end(), item);
                if (it == stack.end()) {
                    stack.push_back(item);
                }
                index++;
            }
        }


        for (auto &it : gradMap) {
            nodes_.push_back(it.second);
            if (it.first->isVariable) {
                variableNodes_[it.first] = it.second;
            }
        }


        // todo:: add optimizer to add applyGradNode
        for (auto &it : variableNodes_) {
            auto applyGradNode = Node::Create();
            applyGradNode->inputs.push_back(it.first);
            applyGradNode->inputs.push_back(it.second);
            applyGradNode->context = it.first->context;
            applyGradNode->opName = "applyGrad";
            applyGradNode->nodeName = it.first->nodeName + "_apply_" + it.second->nodeName;
            applyGradNode->params["learning_rate"] = 0.001f;
            applyGradNode->params["apply_mode"] = kMomentum;
            applyGradNode->params["momentum_factor"] = 0.9f;
            applyGradNode->Build();
            nodes_.push_back(applyGradNode);

        }
    }

    bool Graph::less(const NodePtr &lhs, const NodePtr &rhs) {
        return lhs->id_ < rhs->id_;
    }
}