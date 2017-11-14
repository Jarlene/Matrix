//
// Created by Jarlene on 2017/8/6.
//

#ifndef MATRIX_NODE_H
#define MATRIX_NODE_H
#include <string>
#include <map>
#include <vector>
#include "matrix/include/utils/Any.h"
#include "matrix/include/utils/Registry.h"

namespace matrix {
    struct Node;

    typedef std::weak_ptr<Node> NodeWeakPtr;
    typedef std::shared_ptr<Node> NodePtr;


    struct Node : public std::enable_shared_from_this<Node>{
        Node();

        size_t id_;

        std::string nodeName;

        std::string opName;

        Operator* op{nullptr};

        Context context;

        std::vector<Shape*> inputShapes;

        std::vector<void*> inputDates;

        Shape outputShapes;

        bool isVariable;

        void* data_ {nullptr};

        long memorySize;

        bool isBackward = false;

        bool isPlaceHolder = false;

        std::vector<NodePtr> inputs;

        std::vector<NodeWeakPtr> outputs;

        std::map<std::string, Any> params;

        void SetData();

        long getMemorySize();

        void Build();
        NodePtr  GetGradNode(int input_index, NodePtr &pre, NodePtr &preGrad);
        static NodePtr Create();

    };
}



#endif //MATRIX_NODE_H
