//
// Created by 郑珊 on 2017/8/6.
//

#ifndef MATRIX_NODE_H
#define MATRIX_NODE_H
#include <string>
#include <unordered_map>
#include <vector>
#include "matrix/include/utils/Any.h"
#include "matrix/include/utils/Registry.h"

namespace matrix {
    struct Node;

    typedef std::weak_ptr<Node> NodeWeakPtr;
    typedef std::shared_ptr<Node> NodePtr;


    struct Node {
        Node();

        size_t id_;

        std::string opName;

        OpPtr op_;

        bool isBackward = false;

        std::vector<NodePtr> inputs;

        std::vector<NodeWeakPtr> outputs;

        std::unordered_map<std::string, Any> params;

        void Build();
        NodePtr  GetGradNode(int input_index, NodePtr &pre, NodePtr &preGrad);
        static NodePtr Create();

    };
}



#endif //MATRIX_NODE_H
