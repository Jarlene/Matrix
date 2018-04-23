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


    /**
     *   a&~b:   清除标志位b;
     *    a|b:   添加标志位b;
     *    a&b:   取出标志位b;
     *    a^b:   取出a与b的不同部分;
     */
    const static int VARIABLE_FLAG = 0x00000001;
    const static int PLACEHOLDER_FLAG = VARIABLE_FLAG << 1;
    const static int SHARED_FLAG = PLACEHOLDER_FLAG << 1;
    const static int BACKWARD_FLAG = SHARED_FLAG << 1;


    struct Node;

    typedef std::weak_ptr<Node> NodeWeakPtr;
    typedef std::shared_ptr<Node> NodePtr;


    struct Node : public std::enable_shared_from_this<Node> {
        Node();

        ~Node();

        size_t id_;

        std::string nodeName = "";

        std::string opName;

        Operator *op{nullptr};

        Context context = Context::Default();

        std::vector<Shape *> inputShapes;

        std::vector<void *> inputDates;

        Shape outputShapes;

        int flags = 0;

        void *data_{nullptr};

        long memorySize;

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
            int size = outputShapes.Size();
            int rank = outputShapes.Rank();
            switch (context.type) {
                case kFloat: {
                    auto data = static_cast<float *>(data_);
                    if (rank == 2) {
                        Logger::PrintMat<float>(data, outputShapes[0], outputShapes[1], nodeName);
                    } else {
                        Logger::PrintMat<float>(data, size/outputShapes[rank -1], outputShapes[rank -1], nodeName);
                    }
                }

                    break;
                case kInt: {
                    auto data = static_cast<int *>(data_);
                    if (rank == 2) {
                        Logger::PrintMat<int>(data, outputShapes[0], outputShapes[1], nodeName);
                    } else {
                        Logger::PrintMat<int>(data,size/outputShapes[rank -1], outputShapes[rank -1], nodeName);
                    }
                }

                    break;
                case kLong: {
                    auto data = static_cast<long *>(data_);
                    if (rank == 2) {
                        Logger::PrintMat<long>(data, outputShapes[0], outputShapes[1], nodeName);
                    } else {
                        Logger::PrintMat<long>(data, size/outputShapes[rank -1], outputShapes[rank -1], nodeName);
                    }
                }
                    break;
                case kDouble: {
                    auto data = static_cast<double *>(data_);
                    if (rank == 2) {
                        Logger::PrintMat<double>(data, outputShapes[0], outputShapes[1], nodeName);
                    } else {
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

        void AddFlag(int flag);

        void RemoveFlag(int flag);

        bool HasVariable();

        bool HasShared();

        bool HasBackward();

        bool HasPlaceHolder();

        void Run();

        void DirectRun();

        void On(const Context &context);

        void Save(std::ofstream &ofs);

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
        volatile int depenCount;
    };
}


#endif //MATRIX_NODE_H
