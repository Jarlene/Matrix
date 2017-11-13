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
        OpPtr opPtr = Registry::Global()->GetOp(this->opName, context.type);
        if (opPtr == nullptr) {
            return;
        }
        for(NodePtr node : this->inputs) {
            inputDates.push_back(node->data_);
            inputShapes.push_back(&node->outputShapes);
        }

        op = opPtr->CreateOperator(this->context, &inputDates, this->data_, &inputShapes, &outputShapes, params);
        bool rebuild = false;
        auto generatorVariableFunc = [this, &rebuild](std::initializer_list<Shape *> shapes) {
            rebuild = true;
            for(auto shape = shapes.begin(); shape != shapes.end(); shape++) {
                if (*shape != nullptr) {
                    NodePtr var = Node::Create();
                    var->opName = "variable";
                    var->nodeName = this->nodeName + "_variable";
                    var->outputShapes.reShape(**shape);
                    var->isVariable = context.phase == TRAIN;
                    var->context.type = context.type;
                    var->params["isTrain"] = context.phase == TRAIN;
                    var->outputs.push_back(std::weak_ptr<Node>(this->shared_from_this()));
                    var->Build();
                    this->inputs.push_back(var);
                }
            }

        };
        op->VariableNode(generatorVariableFunc);
        memorySize = opPtr->GetMemorySize();
        if (rebuild) {
            delete  this->op;
            this->op = nullptr;
            this->inputShapes.clear();
            Build();
        }
    }

    long Node::getMemorySize() {
        return memorySize;
    }

}