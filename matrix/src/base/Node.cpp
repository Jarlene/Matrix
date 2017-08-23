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
        t->opName = "grad_" + this->opName;
        t->nodeName = "grad_" + this->nodeName;
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
        t->Build();
        return t;
    }

    void Node::Build() {
        OpPtr opPtr = Registry::Global()->GetOp(this->opName, context.type);
        if (opPtr == nullptr) {
            return;
        }
        std::vector<Blob> inputs;
        std::vector<Blob> outputs;
        for(NodePtr node : this->inputs) {
            inputs.push_back(Blob(node->data_));
            inputShapes.push_back(node->outputShapes);
        }
        std::vector<Shape*> out;
        out.push_back(&outputShapes);
        op = opPtr->CreateOperator(this->context, inputs, outputs, inputShapes, out, params);
        memorySize = opPtr->GetMemorySize();
    }

    long Node::getMemorySize() {
        return memorySize;
    }

}