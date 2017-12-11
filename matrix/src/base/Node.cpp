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
        depenList.clear();
        GET_REGISTRY_OP_PROPERTY(this->opName, context.type);
        if (opPtr == nullptr) {
            return;
        }
        for(NodePtr node : this->inputs) {
            inputShapes.push_back(&node->outputShapes);
        }
        op = opPtr->CreateOperator(&context, &inputShapes, &outputShapes, params);
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
        depenList.insert(depenList.end(), inputs.begin(), inputs.end());
        depenList.sort();
        depenList.unique();
        this->depens = static_cast<int>(depenList.size());
        memorySize = opPtr->GetMemorySize();
    }

    long Node::GetMemorySize() {
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

    void Node::AddInput(const NodePtr &node) {
        this->inputs.push_back(node);
        node->outputs.push_back(std::weak_ptr<Node>(this->shared_from_this()));
    }

    void Node::AddOutput(const NodePtr &node) {
        this->outputs.push_back(std::weak_ptr<Node>(node));
        node->inputs.push_back(this->shared_from_this());
    }

    bool Node::less(const NodePtr &lhs, const NodePtr &rhs) {
        return lhs->id_ < rhs->id_;
    }

    bool Node::large(const NodePtr &lhs, const NodePtr &rhs) {
        return lhs->id_ > rhs->id_;
    }

    std::string Node::ToString() {
        return nodeName + "_" + std::to_string(id_);
    }

    void Node::AddParam(const std::string &name, const Any &any) {
        this->params[name] = any;
    }

    void Node::Complete() {
        for (auto &node : outputs) {
            node.lock()->CountDown();
        }
        this->depens = static_cast<int>(depenList.size());
    }

    void Node::Run() {
        if (this->op != nullptr && !isPlaceHolder && !isShared) {
            if (depens > 0) {
                Await();
            }
            SetData();
            op->AsyncRun();
        }
        Complete();
    }

    void Node::CountDown() {
        std::lock_guard<std::mutex> lock(mutex);
        --depens;
        if (depens <= 0) {
            condvar.notify_all();
        }

    }

    void Node::Await() {
        std::unique_lock<std::mutex> lock(mutex);
        condvar.wait(lock, [this]{ return this->depens <= 0;});
    }

    void Node::AddOpName(const std::string &op) {
        this->opName = op;
    }

    void Node::SwitchType(const Context &context) {
        if (this->context.type == kInvalid) {
            this->context.type = context.type;
        }
    }

    void Node::AddNodeName(const std::string &nodeName) {
        this->nodeName = nodeName;
    }

    void Node::DirectRun() {
        if (this->op != nullptr && !isPlaceHolder && !isShared) {
            SetData();
            op->AsyncRun();
        }
    }

    void Node::Reset() {
        depens = depenList.size();
    }

}