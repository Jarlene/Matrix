//
// Created by Jarlene on 2017/11/15.
//

#include <matrix/include/base/Node.h>
#include <matrix/include/executor/DAGGraph.h>
using namespace matrix;


int main(int argc, char *argv[]) {

    auto n0 = DAGNode<NodePtr>(Node::Create());
    auto n1 = DAGNode<NodePtr>(Node::Create());
    auto n2 = DAGNode<NodePtr>(Node::Create());
    auto n3 = DAGNode<NodePtr>(Node::Create());
    auto n4 = DAGNode<NodePtr>(Node::Create());
    auto n5 = DAGNode<NodePtr>(Node::Create());
    auto n6 = DAGNode<NodePtr>(Node::Create());

    n0.addChild(n1);
    n1.addChild(n2);
    n1.addChild(n3);
    n3.addChild(n4);
    n3.addChild(n6);
    n4.addChild(n5);
    n5.addChild(n6);

    BFSRecurseVisitor<DAGNode<NodePtr>> bfs;

    std::cout << std::endl << "TRAVERSE CHILDREN (start Node 0)  1 level" << std::endl;
    for (auto n : bfs.traverseChildren(n0, 3)) {
        std::cout << n->value()->id_  << std::endl;
    }

    std::cout << std::endl << "TRAVERSE UNDIRECTED (start Node 5) 2 levels " << std::endl;
    for (auto n : bfs.traverseUndirected(n5, 5)) {
        std::cout << n->value()->id_ << std::endl;
    }

    std::cout << std::endl << "TRAVERSE CHILDREN (start Node 0)  all levels" << std::endl;

    for (auto n : bfs.traverseParents(n5)) {
        std::cout << n->value()->id_ << std::endl;
    }
    return 0;
}