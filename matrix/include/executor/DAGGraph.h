//
// Created by Jarlene on 2017/11/17.
//

#ifndef MATRIX_DAGGRAPH_H
#define MATRIX_DAGGRAPH_H

#include <queue>
#include <vector>
#include <unordered_set>

namespace matrix {

    template<typename T>
    class DAGNode {
    public:
        typedef DAGNode<T> TNode;

        DAGNode(const T &v) : m_val(v) {};

        DAGNode();

        TNode &operator=(TNode &) = delete;

        TNode &operator=(const TNode &) = delete;

        DAGNode(TNode &) = default;

        DAGNode(const TNode &) = default;

        TNode &operator=(TNode &&other) = default;

        DAGNode(TNode &&other) = default;

        const T &value() const { return m_val; };

        const std::unordered_set<const TNode *> &children() const { return m_children; }

        const std::unordered_set<const TNode *> &parents() const { return m_parents; }

        void addChild(DAGNode &node) {
            m_children.insert(&node);
            node.addParent(*this);
        }

        void addParent(DAGNode &node) { m_parents.insert(&node); }

    protected:
        T m_val;
        std::unordered_set<const TNode *> m_children;
        std::unordered_set<const TNode *> m_parents;
    };

    template<typename N>
    class Visitor {
        virtual void visit(const N *node) = 0;

        virtual const std::vector<const N *> &traverseChildren(const N &startnode, int depth) = 0;

        virtual const std::vector<const N *> &traverseParents(const N &startnode, int depth) = 0;

        virtual const std::vector<const N *> &traverseUndirected(const N &startnode, int depth) = 0;
    };

    template<typename N>
    class BFSVisitor : public Visitor<N> {
    public:
        BFSVisitor() {};

        void visit(const N *node) override {
            result.push_back(node);  // add to result
            visited.insert(node);    // mark it as visited
        }

        const std::vector<const N *> &traverseChildren(const N &node, int depth = -1) override {
            result = {};                // reset the list of results:
            std::unordered_set<const N *> root{&node};  // create an initial nodeset containing the root node
            traverse(root, BFSVisitor<N>::enumVisitType::CHILDREN, depth);
            visited = {};  // reset the list of visited nodes
            return result;
        }

        const std::vector<const N *> &traverseParents(const N &node, int depth = -1) override {
            result = {};                // reset the list of results
            std::unordered_set<const N *> root{&node};  // create an initial nodeset containing the root node
            traverse(root, BFSVisitor<N>::enumVisitType::PARENTS, depth);
            visited = {};  // reset the list of visited nodes
            return result;
        }

        const std::vector<const N *> &traverseUndirected(const N &node, int depth = -1) override {
            result = {};                // reset the list of results
            std::unordered_set<const N *> root{&node};  // create an initial nodeset containing the root node
            traverse(root, BFSVisitor<N>::enumVisitType::UNDIRECTED, depth);
            visited = {};  // reset the list of visited nodes
            return result;
        }

    protected:
        std::unordered_set<const N *> visited;
        std::vector<const N *> result;
        enum class enumVisitType {
            CHILDREN = 0,
            PARENTS,
            UNDIRECTED
        };

        virtual void traverse(const std::unordered_set<const N *> &nodes, BFSVisitor<N>::enumVisitType visittype,
                              int depth) {
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

        bool alreadyVisited(const N *node) const {
            return !(visited.find(node) == visited.end());
        }
    };

    template<typename N>
    class BFSRecurseVisitor : public BFSVisitor<N> {
    public:
    private:
        virtual void
        traverse(const std::unordered_set<const N *> &nodes, typename BFSVisitor<N>::enumVisitType visittype,
                 int depth) override {
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
    };


    template<typename N>
    class DFSVisitor : public Visitor<N> {
    public:
        DFSVisitor();

        void visit(const N *node) override {

        }

        const std::vector<const N *> &traverseChildren(const N &node, int depth = -1) override {
            return std::vector<const N *>();
        }

        const std::vector<const N *> &traverseParents(const N &node, int depth = -1) override {
            return std::vector<const N *>();
        }

        const std::vector<const N *> &traverseUndirected(const N &node, int depth = -1) override {
            return std::vector<const N *>();
        }

    protected:
        std::unordered_set<const N *> visited;
        std::vector<const N *> result;
        enum class enumVisitType {
            CHILDREN = 0,
            PARENTS,
            UNDIRECTED
        };

        virtual void traverse(const std::unordered_set<const N *> &nodes, DFSVisitor<N>::enumVisitType visittype,
                              int depth) {

        }

        bool alreadyVisited(const N *node) const {
            return false;
        }
    };


}

#endif //MATRIX_DAGGRAPH_H
