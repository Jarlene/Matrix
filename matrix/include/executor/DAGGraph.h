//
// Created by Jarlene on 2017/11/17.
//

#ifndef MATRIX_DAGGRAPH_H
#define MATRIX_DAGGRAPH_H

#include <vector>
#include <unordered_set>

namespace matrix {

    template <typename N>
    class Visitor {
        virtual void visit(const N *node) = 0;

        virtual const std::vector <const N*> &traverseChildren(const N &startnode, int depth) = 0;

        virtual const std::vector <const N*> &traverseParents(const N &startnode, int depth) = 0;

        virtual const std::vector <const N*> &traverseUndirected(const N &startnode, int depth) = 0;
    };

    template <typename N>
    class BFSVisitor : public Visitor<N> {
    public:
        BFSVisitor();
        void visit(const N* node) override;
        const std::vector <const N*>& traverseChildren(const N& node, int depth = -1) override;
        const std::vector <const N*>& traverseParents(const N& node, int depth = -1) override;
        const std::vector <const N*>& traverseUndirected(const N& node, int depth = -1) override;
    protected:
        std::unordered_set<const N*> visited;
        std::vector <const N*> result;
        enum class enumVisitType {
            CHILDREN = 0,
            PARENTS,
            UNDIRECTED
        };
        virtual void traverse(const std::unordered_set<const N*>& nodes, BFSVisitor<N>::enumVisitType visittype,
                              int depth);
        bool alreadyVisited(const N* node) const;
    };

    template <typename N>
    class BFSRecurseVisitor : public BFSVisitor<N> {
    public:
    private:
        virtual void traverse(const std::unordered_set<const N*>& nodes, typename BFSVisitor<N>::enumVisitType visittype,
                              int depth) override;
    };

    class DAGGraph {

    };

}

#endif //MATRIX_DAGGRAPH_H
