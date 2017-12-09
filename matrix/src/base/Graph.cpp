//
// Created by Jarlene on 2017/8/9.
//

#include "matrix/include/executor/GraphAlgorithm.h"
#include "matrix/include/base/Graph.h"
#include "matrix/include/store/MemoryManager.h"

namespace matrix {



    static void addNode(std::vector<NodePtr> &stack, NodePtr &node) {
        stack.push_back(node);
        for (auto &item : node->inputs) {
            stack.push_back(item);
            addNode(stack, item);
        }
    }

    static NodePtr AddGrad(const NodePtr &first, const NodePtr &second) {
        auto res = Node::Create();
        res->opName = "add";
        res->nodeName = first->nodeName + "_add_" + second->nodeName;
        res->isBackward = false;
        res->inputs.push_back(first);
        res->inputs.push_back(second);
        first->outputs.push_back(std::weak_ptr<Node>(res));
        second->outputs.push_back(std::weak_ptr<Node>(res));
        res->Build();
        return res;
    }


    Graph::Graph(const Symbol &symbol, BaseOptimizer* optimizer,  bool isTrain) : optimizer(optimizer), isTrain(isTrain) {
        if (isTrain) {
            GeneratorGradNodes(symbol);
        }
    }

    Graph::~Graph() {
        delete optimizer;
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

    void Graph::AllocateGraph() {
        for(auto &node : nodes_) {
            if (node->op != nullptr && node->memorySize > 0
                && node->data_ == nullptr && !node->isPlaceHolder) {
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
//                if (maxSize < node->GetMemorySize()) {
//                    maxSize = node->GetMemorySize();
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
//                size_t m = item->GetMemorySize();
//                if (m > 0) {
//                    item->data_ = MemoryManager::Global()->GetCpuMemoryPool()->dynamicAllocate(m);
//                } else {
//                    Logger::Global()->Info("%s not in memory \n", item->nodeName.c_str());
//                }
//
//            }
//        }
        MemoryManager::Global()->GetCpuMemoryPool()->PrintMemory();
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
        sort(nodes_.begin(), nodes_.end(), Node::less);
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
        ones->isBackward = true;
        ones->params["isTrain"] = true;
        ones->params["constant"] = 1.0f;
        ones->Build();
        gradMap[out] = ones;


        std::vector<NodePtr> forward;
        addNode(forward, out);
        sort(forward.begin(), forward.end(), Node::less);
        forward.erase(unique(forward.begin(), forward.end()), forward.end());
        nodes_.insert(nodes_.end(), forward.begin(), forward.end());
        this->forward.insert(this->forward.end(), forward.begin(), forward.end());
        while (!forward.empty()) {
            auto pre = forward.back();
            forward.pop_back();
            int index = 0;
            for (auto &item : pre->inputs) {
                auto grad_node = item->GetGradNode(index, pre, gradMap[pre]);
                if (grad_node == nullptr) {
                    continue;
                }
                if (gradMap.count(item)) {
                    nodes_.push_back(gradMap[item]);
                    nodes_.push_back(grad_node);
                    auto add = AddGrad(gradMap[item], grad_node);
                    gradMap[item] = add;
                } else {
                    gradMap[item] = grad_node;
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

        if (optimizer != nullptr) {
            auto apply = optimizer-> GeneratorUpdate(variableNodes_);
            variables.insert(variables.end(), apply.begin(), apply.end());
        }
    }

    const std::vector<NodePtr> &Graph::GetUpdateNodes() const {
        return variables;
    }

    const std::vector<NodePtr> &Graph::GetForwardNodes() const {
        return forward;
    }

    NodePtr Graph::Accuracy(const Symbol *symbol) {
        auto node = symbol->GetNode();
        if (node->data_ == nullptr) {
            node->data_ = MemoryManager::Global()->GetCpuMemoryPool()->dynamicAllocate(node->memorySize);
        }
        return node;
    }

    void* Graph::Evaluating(const Symbol *symbol) {
        auto node = symbol->GetNode();
        if (node->data_ == nullptr) {
            node->data_ = MemoryManager::Global()->GetCpuMemoryPool()->dynamicAllocate(node->memorySize);
        }
        node->Run();
        return node->data_;
    }

}