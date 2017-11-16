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
        Graph(const Symbol &symbol, BaseOptimizer* optimizer,  bool isTrain);

        ~Graph();

        NodePtr GetNode(const std::string &name);

        NodePtr GetNode(size_t id);

        void Optimize();

        void AllocateGraph();

        void SaveVariableData(std::string &file);

        void SaveModel(std::string &file);

        void AppendNode(const NodePtr &node);

        const std::vector<NodePtr> &GetGraphNodes() const;

        const std::vector<NodePtr> &GetUpdateNodes() const ;
    private:


        void Unique();

        void GeneratorGradNodes(const Symbol &symbol);

    private:
        std::map<int, int> graphColor_;
        std::vector<NodePtr> nodes_;
        std::vector<NodePtr> variables;
        /// the first is variable, the second is grad_variabl
        std::map<NodePtr, NodePtr> variableNodes_;
        BaseOptimizer *optimizer;
        bool isTrain;
    };


}

#endif //MATRIX_GRAPH_H
