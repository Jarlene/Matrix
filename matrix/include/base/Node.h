//
// Created by Jarlene on 2017/8/6.
//

#ifndef MATRIX_NODE_H
#define MATRIX_NODE_H

#include <string>
#include <map>
#include <vector>
#include <list>
#include <sstream>
#include "matrix/include/utils/Any.h"
#include "matrix/include/utils/Registry.h"

namespace matrix {
    struct Node;

    typedef std::weak_ptr<Node> NodeWeakPtr;
    typedef std::shared_ptr<Node> NodePtr;


    struct Node : public std::enable_shared_from_this<Node> {
        Node();

        size_t id_;

        std::string nodeName = "";

        std::string opName;

        Operator *op{nullptr};

        Context context = Context::Default();

        std::vector<Shape *> inputShapes;

        std::vector<void *> inputDates;

        Shape outputShapes;

        bool isVariable;

        bool isShared = false;

        void *data_{nullptr};

        long memorySize;

        bool isBackward = false;

        bool isPlaceHolder = false;

        std::vector<NodePtr> inputs;

        std::list<NodePtr> depenList;

        std::vector<NodeWeakPtr> outputs;

        std::map<std::string, Any> params;

        void AddOpName(const std::string &op);

        void AddNodeName(const std::string &nodeName);

        long GetMemorySize();

        void Build();

        NodePtr GetGradNode(int input_index, NodePtr &pre, NodePtr &preGrad);

        static NodePtr Create();

        bool operator==(const NodePtr &node);

        bool operator<(const NodePtr &node);

        static bool less(const NodePtr &lhs, const NodePtr &rhs);

        static bool large(const NodePtr &lhs, const NodePtr &rhs);

        void PrintMatrix() {
            if (data_ == nullptr) {
                return;
            }
            switch (context.type) {
                case kFloat: {
                    auto data = static_cast<float *>(data_);
                    if (outputShapes.Rank() == 2) {
                        Logger::PrintMat<float>(data, outputShapes[0], outputShapes[1], nodeName);
                    } else {
                        int size = outputShapes.Size();
                        int rank = outputShapes.Rank();
                        Logger::PrintMat<float>(data, size/outputShapes[rank -1], outputShapes[rank -1], nodeName);
                    }
                }

                    break;
                case kInt: {
                    auto data = static_cast<int *>(data_);
                    if (outputShapes.Rank() == 2) {
                        Logger::PrintMat<int>(data, outputShapes[0], outputShapes[1], nodeName);
                    } else {
                        int size = outputShapes.Size();
                        int rank = outputShapes.Rank();
                        Logger::PrintMat<int>(data,size/outputShapes[rank -1], outputShapes[rank -1], nodeName);
                    }
                }

                    break;
                case kLong: {
                    auto data = static_cast<long *>(data_);
                    if (outputShapes.Rank() == 2) {
                        Logger::PrintMat<long>(data, outputShapes[0], outputShapes[1], nodeName);
                    } else {
                        int size = outputShapes.Size();
                        int rank = outputShapes.Rank();
                        Logger::PrintMat<long>(data, size/outputShapes[rank -1], outputShapes[rank -1], nodeName);
                    }
                }
                    break;
                case kDouble: {
                    auto data = static_cast<double *>(data_);
                    if (outputShapes.Rank() == 2) {
                        Logger::PrintMat<double>(data, outputShapes[0], outputShapes[1], nodeName);
                    } else {
                        int size = outputShapes.Size();
                        int rank = outputShapes.Rank();
                        Logger::PrintMat<double>(data, size/outputShapes[rank -1], outputShapes[rank -1], nodeName);
                    }
                }
                    break;
                default:
                    Logger::Global()->Error("can not print data");
                    break;
            }
        }

        std::string ToString();

        void AddInput(const NodePtr &node);

        void AddOutput(const NodePtr &node);

        void AddParam(const std::string &name, const Any &any);

        void Run();

        void DirectRun();

        void SwitchType(const Context &context);

    private:
        friend class Graph;
        void Reset();

    private:
        void SetData();
        void CountDown();
        void Complete();
        void Await();

    private:
        std::mutex mutex;
        std::condition_variable condvar;
        std::atomic<int> depens;
    };
}


#endif //MATRIX_NODE_H
