//
// Created by Jarlene on 2017/8/9.
//

#ifndef MATRIX_GRAPH_H
#define MATRIX_GRAPH_H

#include <string>
#include "matrix/include/api/Symbol.h"
#include "matrix/include/base/Node.h"
#include "matrix/include/optimizer/BaseOptimizer.h"

namespace matrix {


    class Graph {
    public:
        Graph(const Symbol &symbol,  bool isTrain);

        ~Graph();

        NodePtr GetNode(const std::string &name);

        NodePtr GetNode(size_t id);

        void Optimize();

        void AllocateGraph(const std::vector<NodePtr> &fetch);

        void SaveVariableData(std::string &file);

        void SaveModel(std::string &file);

        void AppendNode(const NodePtr &node);

        const std::vector<NodePtr> &GetGraphNodes() const;

    private:

        static bool less(const NodePtr &lhs, const NodePtr &rhs);
        void Unique();

        void GeneratorGradNodes(const Symbol &symbol);

    private:
        std::map<int, int> graphColor_;
        std::vector<NodePtr> nodes_;
        /// the first is variable, the second is grad_variabl
        std::map<NodePtr, NodePtr> variableNodes_;
    };


}

#endif //MATRIX_GRAPH_H
