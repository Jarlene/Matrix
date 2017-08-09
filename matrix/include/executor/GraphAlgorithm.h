//
// Created by Jarlene on 2017/8/9.
//

#ifndef MATRIX_GRAPHALGORITHM_H
#define MATRIX_GRAPHALGORITHM_H

#include "matrix/include/base/Graph.h"
#include "matrix/include/base/Node.h"


namespace matrix {
    class GraphAlgorithm {
    public:

        GraphAlgorithm();

        void Coloring(Graph &graph, std::map<int, int> &colors);

        void Coloring(Graph &graph, std::map<int, int> &colors, const std::vector<NodePtr> &fetch);

    private:

        static bool compare(const NodePtr &lhs, const NodePtr &rhs) {
            return lhs->id_ < rhs->id_;
        }

    private:
        std::vector<NodePtr> fetchNodes;
    };
}

#endif //MATRIX_GRAPHALGORITHM_H
