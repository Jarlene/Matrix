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


    struct Node : public std::enable_shared_from_this<Node>{
        Node();

        size_t id_;

        std::string nodeName = "";

        std::string opName;

        Operator* op{nullptr};

        Context context;

        std::vector<Shape*> inputShapes;

        std::vector<void*> inputDates;

        Shape outputShapes;

        bool isVariable;

        bool isShared = false;

        void* data_ {nullptr};

        long memorySize;

        bool isBackward = false;

        bool isPlaceHolder = false;

        std::vector<NodePtr> inputs;

        std::vector<NodeWeakPtr> outputs;

        std::list<NodePtr> depens_;

        std::map<std::string, Any> params;

        void SetData();

        long getMemorySize();

        void Build();
        NodePtr  GetGradNode(int input_index, NodePtr &pre, NodePtr &preGrad);
        static NodePtr Create();

        bool operator==(const NodePtr &node);

        bool operator<(const NodePtr &node);

        static bool less(const NodePtr &lhs, const NodePtr &rhs) {
            return lhs->id_ < rhs->id_;
        }

        static bool large(const NodePtr &lhs, const NodePtr &rhs) {
            return lhs->id_ > rhs->id_;
        }

        void PrintMatrix() {
//            if (outputShapes.Rank() == 2) {
//                std::ostringstream stream;
//                std::ofstream ofile;
//                ofile.open("/Users/jarlene/Code/Cpp/AI/Matrix/" + this->ToString() + ".txt");
//                for (int i = 0; i < outputShapes[0]; ++i) {
//                    for (int j = 0; j < outputShapes[1]; ++j) {
//                        stream << (static_cast<float*>(data_))[i * outputShapes[1] + j] << "     ";
//                    }
//                    stream << "\n";
//                }
//
//                ofile << stream.str();
//                ofile.close();
//            } else if (outputShapes.Rank() == 1) {
//                std::ostringstream stream;
//                std::ofstream ofile;
//                ofile.open("/Users/jarlene/Code/Cpp/AI/Matrix/" + this->ToString() + ".txt");
//                for (int i = 0; i < outputShapes[0]; ++i) {
//                    stream << (static_cast<float*>(data_))[i] << "     " << "\n";
//                }
//                ofile << stream.str();
//                ofile.close();
//            }


            switch(context.type) {
                case kFloat:
                {
                    auto data = static_cast<float*>(data_);
                    if (outputShapes.Rank() == 2) {
                        Logger::PrintMat<float>(data, outputShapes[0], outputShapes[1], nodeName);
                    } else {
                        Logger::PrintMat<float>(data, outputShapes[0], 1, nodeName);
                    }
                }

                    break;
                case kInt:
                {
                    auto data = static_cast<int*>(data_);
                    if (outputShapes.Rank() == 2) {
                        Logger::PrintMat<int>(data, outputShapes[0], outputShapes[1], nodeName);
                    } else {
                        Logger::PrintMat<int>(data, outputShapes[0], 1, nodeName);
                    }
                }

                    break;
                case kLong:
                {
                    auto data = static_cast<long*>(data_);
                    if (outputShapes.Rank() == 2) {
                        Logger::PrintMat<long>(data, outputShapes[0], outputShapes[1], nodeName);
                    } else {
                        Logger::PrintMat<long>(data, outputShapes[0], 1, nodeName);
                    }
                }
                    break;
                case kDouble:
                {
                    auto data = static_cast<double*>(data_);
                    if (outputShapes.Rank() == 2) {
                        Logger::PrintMat<double>(data, outputShapes[0], outputShapes[1], nodeName);
                    } else {
                        Logger::PrintMat<double>(data, outputShapes[0], 1, nodeName);
                    }
                }
                    break;
                default:
                    Logger::Global()->Error("can not print data");
                    break;
            }
        }


        std::string ToString() {
            return nodeName + "_" + std::to_string(id_);
        }

    };
}



#endif //MATRIX_NODE_H
