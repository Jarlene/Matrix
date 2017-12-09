//
// Created by Jarlene on 2017/11/15.
//

#include <matrix/include/base/Node.h>
#include <matrix/include/executor/DAGGraph.h>
using namespace matrix;


int main(int argc, char *argv[]) {

    auto n0 = Node::Create();
    auto n1 = Node::Create();
    auto n2 = Node::Create();
    auto n3 = Node::Create();
    auto n4 = Node::Create();
    auto n5 = Node::Create();
    auto n6 = Node::Create();

    n0->AddOutput(n1);
    n1->AddOutput(n2);
    n1->AddOutput(n3);
    n3->AddOutput(n4);
    n3->AddOutput(n6);
    n4->AddOutput(n5);
    n5->AddOutput(n6);

    BFSRecurseVisitor<NodePtr> bfs;

    std::cout << std::endl << "TRAVERSE CHILDREN (start Node 0)  1 level" << std::endl;
    for (auto n : bfs.traverseChildren(n0, 1)) {
        std::cout << n->get()->id_  << std::endl;
    }

    std::cout << std::endl << "TRAVERSE UNDIRECTED (start Node 5) 2 levels " << std::endl;
    for (auto n : bfs.traverseUndirected(n5, 2)) {
        std::cout << n->get()->id_ << std::endl;
    }

    std::cout << std::endl << "TRAVERSE CHILDREN (start Node 0)  all levels" << std::endl;

    for (auto n : bfs.traverseChildren(n0)) {
        std::cout << n->get()->id_ << std::endl;
    }
    return 0;
}