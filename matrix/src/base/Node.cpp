//
// Created by Jarlene on 2017/8/6.
//

#include "matrix/include/base/Node.h"

namespace matrix {
    static size_t index = 0;

    NodePtr Node::Create() {
        return std::make_shared<Node>();
    }



    Node::Node() {
        id_ = index ++;
    }

    NodePtr Node::GetGradNode(int input_index, NodePtr &pre, NodePtr &preGrad) {
        if(this->isShared) {
            return nullptr;
        }
        auto t = Node::Create();
        t->isBackward = true;
        t->opName = "grad_" + pre->opName;
        t->nodeName = "grad_" + pre->nodeName;
        t->params["input_idx"] = input_index;
        t->context = this->context;
        t->inputs.push_back(preGrad);
        t->inputs.push_back(pre);
        pre->outputs.push_back(std::weak_ptr<Node>(t));
        preGrad->outputs.push_back(std::weak_ptr<Node>(t));
        for(NodePtr ptr : pre->inputs) {
            t->inputs.push_back(ptr);
            ptr->outputs.push_back(std::weak_ptr<Node>(t));
        }
        for (auto &it : pre->params) {
            t->params.insert(it);
        }
        if (this->isVariable) {
            t->isVariable = true;
        }
        if (this->isPlaceHolder) {
            t->isPlaceHolder = true;
        }
        t->Build();
        return t;
    }

    void Node::Build() {

        GET_REGISTRY_OP_PROPERTY(this->opName, context.type);
        if (opPtr == nullptr) {
            return;
        }
        for(NodePtr node : this->inputs) {
            inputShapes.push_back(&node->outputShapes);
        }
        op = opPtr->CreateOperator(this->context, &inputShapes, &outputShapes, params);
        if (op == nullptr) {
            Logger::Global()->Fatal("can not find the op %s", this->opName.c_str());
        }
        auto generatorVariableFunc = [this](std::initializer_list<Shape *> shapes) {
            int idx = 0;
            for(auto shape = shapes.begin(); shape != shapes.end(); shape++) {
                if (*shape != nullptr) {
                    NodePtr var = Node::Create();
                    var->opName = "variable";
                    var->nodeName = this->nodeName + "_variable_" + std::to_string(idx);
                    var->outputShapes.reShape(**shape);
                    var->isVariable = context.phase == TRAIN;
                    var->context.type = context.type;
                    var->params["isTrain"] = context.phase == TRAIN;
                    var->outputs.push_back(std::weak_ptr<Node>(this->shared_from_this()));
                    var->Build();
                    this->inputs.push_back(var);
                    idx++;
                }
            }

        };
        auto generatorSharedFunc = [this](std::initializer_list<Shape *> shapes) {
            int idx = 0;
            for(auto shape = shapes.begin(); shape != shapes.end(); shape++) {
                if (*shape != nullptr) {
                    NodePtr var = Node::Create();
                    var->opName = "variable";
                    var->nodeName = this->nodeName + "_shared_" + std::to_string(idx);
                    var->outputShapes.reShape(**shape);
                    var->isVariable = false;
                    var->isShared = true;
                    var->context.type = context.type;

                    var->outputs.push_back(std::weak_ptr<Node>(this->shared_from_this()));
                    var->Build();
                    this->inputs.push_back(var);
                    idx++;
                }
            }
        };
        if (op != nullptr) {
            bool rebuild = op->VariableNode(generatorVariableFunc);
            if (rebuild) {
                delete  this->op;
                this->op = nullptr;
                this->inputShapes.clear();
                Build();
            } else {
                bool shared = op->ShareNodes(generatorSharedFunc);
                if (shared) {
                    delete  this->op;
                    this->op = nullptr;
                    this->inputShapes.clear();
                    Build();
                }
            }
        }
        memorySize = opPtr->GetMemorySize();
    }

    long Node::getMemorySize() {
        return memorySize;
    }

    void Node::SetData() {
        inputDates.clear();
        for (auto it = inputs.begin(); it != inputs.end(); it++) {
            inputDates.push_back((*it)->data_);
        }
        op->SetData(&inputDates, data_);
    }

    bool Node::operator==(const NodePtr &node) {
        return this->id_ == node->id_;
    }

    bool Node::operator<(const NodePtr &node) {
        return this->id_ < node->id_;
    }

}