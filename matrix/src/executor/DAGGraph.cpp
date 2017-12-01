//
// Created by Jarlene on 2017/11/17.
//

#include <assert.h>
#include <queue>
#include "matrix/include/executor/DAGGraph.h"

namespace matrix {

    template<typename N>
    void BFSVisitor<N>::visit(const N *node) {
        result.push_back(node);  // add to result
        visited.insert(node);    // mark it as visited
    }

    template<typename N>
    const std::vector<const N *> &BFSVisitor<N>::traverseChildren(const N &node, int depth) {
        result = {};                // reset the list of results:
        std::unordered_set<const N *> root{&node};  // create an initial nodeset containing the root node
        traverse(root, BFSVisitor<N>::enumVisitType::CHILDREN, depth);
        visited = {};  // reset the list of visited nodes
        return result;
    }

    template<typename N>
    const std::vector<const N *> &BFSVisitor<N>::traverseParents(const N &node, int depth) {
        result = {};                // reset the list of results
        std::unordered_set<const N *> root{&node};  // create an initial nodeset containing the root node
        traverse(root, BFSVisitor<N>::enumVisitType::PARENTS, depth);
        visited = {};  // reset the list of visited nodes
        return result;
    }

    template<typename N>
    const std::vector<const N *> &BFSVisitor<N>::traverseUndirected(const N &node, int depth) {
        result = {};                // reset the list of results
        std::unordered_set<const N *> root{&node};  // create an initial nodeset containing the root node
        traverse(root, BFSVisitor<N>::enumVisitType::UNDIRECTED, depth);
        visited = {};  // reset the list of visited nodes
        return result;
    }

    template<typename N>
    void BFSVisitor<N>::traverse(const std::unordered_set<const N *> &nodes,
                                 BFSVisitor<N>::enumVisitType visittype, int depth) {
        typedef typename BFSVisitor<N>::enumVisitType pt;

        std::queue<const N *> nodeQueue;
        std::queue<int> nodeDepth;

        for (auto const &node : nodes) {
            if (visited.find(node) == visited.end()) {  // if node is not listed as already being visited
                this->visit(node);                      // mark as visited and add to results
                nodeQueue.push(node);                         // put into the queue
                nodeDepth.push(0);
            }
        }

        while (!nodeQueue.empty()) {
            int curdepth = nodeDepth.front();
            if ((depth < 0 || curdepth < depth) &&// NB depth=-1 means we are visiting everything
                ((visittype == pt::CHILDREN) | (visittype == pt::UNDIRECTED))) {  // use the children
                for (auto node : nodeQueue.front()->children()) {
                    if (visited.find(node) == visited.end()) {  // check node is not already being visited
                        this->visit(node);
                        nodeQueue.push(node);
                        nodeDepth.push(curdepth + 1);
                    }
                }
            }
            if ((depth < 0 || curdepth < depth) && //NB depth=-1 means we are visiting everything
                ((visittype == pt::PARENTS) | (visittype == pt::UNDIRECTED))) {  // use the parents
                for (auto node : nodeQueue.front()->parents()) {

                    if (visited.find(node) == visited.end()) {  // check node is not already being visited
                        this->visit(node);

                        nodeQueue.push(node);
                        nodeDepth.push(curdepth + 1);
                    }
                }
            }
            nodeQueue.pop();
            nodeDepth.pop();
        }
    }

    template<typename N>
    bool BFSVisitor<N>::alreadyVisited(const N *node) const {
        return !(visited.find(node) == visited.end());
    }

    template<typename N>
    BFSVisitor<N>::BFSVisitor() : Visitor<N>(), visited() {

    }

    template<typename N>
    void BFSRecurseVisitor<N>::traverse(const std::unordered_set<const N *> &nodes,
                                        typename BFSVisitor<N>::enumVisitType visittype, int depth) {
        // For a recursive  breadth first traversal we gather all nodes at the same depth
        typedef typename BFSVisitor<N>::enumVisitType pt;
        std::unordered_set<const N *> visitnextnodes;  // this collects all the nodes at the next "depth"

        if (nodes.empty()) {
            return;  // end of the recursion
        }

        for (auto node : nodes) {

            // Only process a node if not already visited
            if (BFSVisitor<N>::visited.find(node) == BFSVisitor<N>::visited.end()) {
                // this will add the node to the "result" and mark the node as visited
                this->visit(node);

                // Now add in all the children/parent/undirected links for the next depth
                // and store these into visitnextnodes
                // NB depth=-1 means we are visiting everything
                if (depth != 0 && (visittype == pt::CHILDREN | visittype == pt::UNDIRECTED))
                    for (const auto child : node->children()) {
                        if (!this->alreadyVisited(child)) visitnextnodes.insert(child);
                    }
                if (depth != 0 && (visittype == pt::PARENTS | visittype == pt::UNDIRECTED))
                    for (const auto parent : node->parents()) {
                        if (!this->alreadyVisited(parent)) visitnextnodes.insert(parent);
                    }
            }
        }
        depth--;
        traverse(visitnextnodes, visittype, depth);
    }

    template<typename N>
    DFSVisitor<N>::DFSVisitor() : Visitor<N>(), visited() {

    }

    template<typename N>
    void DFSVisitor<N>::visit(const N *node) {

    }

    template<typename N>
    const std::vector<const N *> &DFSVisitor<N>::traverseChildren(const N &node, int depth) {
        return std::vector<const N *>();
    }

    template<typename N>
    const std::vector<const N *> &DFSVisitor<N>::traverseParents(const N &node, int depth) {
        return std::vector<const N *>();
    }

    template<typename N>
    const std::vector<const N *> &DFSVisitor<N>::traverseUndirected(const N &node, int depth) {
        return std::vector<const N *>();
    }

    template<typename N>
    void DFSVisitor<N>::traverse(const std::unordered_set<const N *> &nodes, DFSVisitor<N>::enumVisitType visittype,
                              int depth) {

    }

    template<typename N>
    bool DFSVisitor<N>::alreadyVisited(const N *node) const {
        return false;
    }

}