//
// Created by 郑珊 on 2017/8/6.
//

#include "matrix/include/base/Node.h"

namespace matrix {
    static size_t index = 0;

    NodePtr Node::create() {
        return std::make_shared<Node>();
    }



    Node::Node() {
        id_ = index ++;
    }

    NodePtr Node::GetGradNode(int input_index, NodePtr &pre, NodePtr &preGrad) {
        auto t = Node::create();

        return t;
    }

    void Node::Build() {

    }

}