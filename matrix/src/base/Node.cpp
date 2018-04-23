//
// Created by Jarlene on 2017/8/6.
//
#include <set>
#include <sstream>
#include <thread>
#include <matrix/include/store/MemoryManager.h>
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
        if(this->HasShared()) {
            return nullptr;
        }
        auto t = Node::Create();
        t->AddFlag(BACKWARD_FLAG);
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
        if (this->HasVariable()) {
            t->AddFlag(VARIABLE_FLAG);
        }
        if (this->HasPlaceHolder()) {
            t->AddFlag(PLACEHOLDER_FLAG);
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
            for (auto shape : shapes) {
                if (shape != nullptr) {
                    NodePtr var = Node::Create();
                    var->opName = "variable";
                    var->nodeName = this->nodeName + "_variable_" + std::to_string(idx);
                    var->outputShapes.reShape(*shape);
                    var->AddFlag(VARIABLE_FLAG);
                    var->context.type = this->context.type;
                    var->params["isTrain"] = this->context.phase == TRAIN;
                    var->outputs.push_back(std::weak_ptr<Node>(this->shared_from_this()));
                    var->Build();
                    this->inputs.push_back(var);
                    idx++;
                }
            }
        };
        auto generatorSharedFunc = [this](std::initializer_list<Shape *> shapes) {
            int idx = 0;
            for (auto shape : shapes) {
                if (shape != nullptr) {
                    NodePtr var = Node::Create();
                    var->opName = "variable";
                    var->nodeName = this->nodeName + "_shared_" + std::to_string(idx);
                    var->outputShapes.reShape(*shape);
                    var->AddFlag(SHARED_FLAG);
                    var->context.type = this->context.type;
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
        this->depenCount = static_cast<int>(depenList.size());
        memorySize = opPtr->GetMemorySize();
    }

    long Node::GetMemorySize() {
        return memorySize;
    }

    void Node::SetData() {
        if (inputDates.empty()) {
            inputDates.clear();
            for (auto it = inputs.begin(); it != inputs.end(); it++) {
                inputDates.push_back((*it)->data_);
            }
            op->SetData(&inputDates, data_);
        }
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
        std::stringstream stream;
        stream << "name[" << nodeName << "]:id[" << id_ << "]:backward[" << HasBackward() << "]:placeHolder["
               << HasPlaceHolder() << "]:shared[" << HasShared() << "]:variable[" << HasVariable() << "]";
        return stream.str();
    }

    void Node::AddParam(const std::string &name, const Any &any) {
        this->params[name] = any;
    }

    void Node::Complete() {
        std::set<NodePtr> set;
        for (auto &node : outputs) {
            set.insert(node.lock());
        }
        for (auto node : set) {
            node->CountDown();
        }
        set.clear();
        this->depenCount = depenList.size();
//        Logger::Global()->Info("%s[%d] complete at thread[%d]", this->nodeName.c_str(), this->id_, std::this_thread::get_id());
    }

    void Node::Run() {
        if (this->op != nullptr && !HasPlaceHolder() && !HasShared()) {
            if (depenCount > 0) {
                Await();
            }
            SetData();
            op->AsyncRun();
        }
        Complete();
    }

    void Node::CountDown() {
        std::unique_lock<std::mutex> lock(mutex);
        depenCount--;
        if (depenCount <= 0) {
            condvar.notify_all();
        }

    }

    void Node::Await() {
        std::unique_lock<std::mutex> lock(mutex);
        condvar.wait(lock, [this]{ return this->depenCount <= 0;});
    }

    void Node::AddOpName(const std::string &op) {
        this->opName = op;
    }

    void Node::On(const Context &context) {
        if (this->context.type == kInvalid) {
            this->context.type = context.type;
        }
    }

    void Node::AddNodeName(const std::string &nodeName) {
        this->nodeName = nodeName;
    }

    void Node::DirectRun() {
        if (this->op != nullptr && !HasPlaceHolder() && !HasShared()) {
            SetData();
            op->AsyncRun();
        }
    }

    void Node::Reset() {
        depenCount = depenList.size();
    }

    Node::~Node() {
        if (op != nullptr) {
            delete op;
            op = nullptr;
        }
        inputs.clear();
        depenList.clear();
        outputs.clear();
        inputDates.clear();
        inputShapes.clear();
        params.clear();
    }

    void Node::Save(std::ofstream &ofs) {

    }

    void Node::AddFlag(int flag) {
        this->flags |= flag;

    }

    void Node::RemoveFlag(int flag) {
        this->flags &= ~flag;
    }

    bool Node::HasVariable() {
        return (this->flags & VARIABLE_FLAG) > 0;
    }

    bool Node::HasShared() {
        return (this->flags & SHARED_FLAG) > 0;
    }

    bool Node::HasBackward() {
        return (this->flags & BACKWARD_FLAG) > 0;
    }

    bool Node::HasPlaceHolder() {
        return (this->flags & PLACEHOLDER_FLAG) > 0;
    }


}