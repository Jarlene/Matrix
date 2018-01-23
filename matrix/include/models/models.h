//
// Created by Jarlene on 2017/7/31.
//

#ifndef MATRIX_ALEXNET_H
#define MATRIX_ALEXNET_H

#include "../api/Symbol.h"
#include "../api/VariableSymbol.h"
#include "../base/Shape.h"
#include "../api/MatrixType.h"

namespace matrix {


    Symbol AlexSymbol(Symbol &input,  int class_num) {


        auto conv1 = Symbol("convolution")
                .SetInput("input", input)
                .SetParam("filter", ShapeN(11, 11))
                .SetParam("padding", ShapeN(0, 0))
                .SetParam("stride", ShapeN(4, 4))
                .SetParam("dilate", ShapeN(1, 1))
                .SetParam("filter_num", 96)
                .SetParam("with_bias", true)
                .SetParam("activation_type", kRelu)
                .SetParam("group", 1)
                .Build("conv1");

        auto pool1 = Symbol("pooling")
                .SetInput("conv1", conv1)
                .SetParam("filter", ShapeN(3, 3))
                .SetParam("stride", ShapeN(2, 2))
                .SetParam("type", kMax)
                .Build("pool1");

        auto conv2 = Symbol("convolution")
                .SetInput("data", pool1)
                .SetParam("filter", ShapeN(5, 5))
                .SetParam("padding", ShapeN(2, 2))
                .SetParam("stride", ShapeN(1, 1))
                .SetParam("dilate", ShapeN(1, 1))
                .SetParam("filter_num", 256)
                .SetParam("with_bias", true)
                .SetParam("activation_type", kRelu)
                .SetParam("group", 1)
                .Build("conv2");

        auto pool2 = Symbol("pooling")
                .SetInput("conv2", conv2)
                .SetParam("filter", ShapeN(3, 3))
                .SetParam("stride", ShapeN(2, 2))
                .SetParam("type", kMax)
                .Build("pool2");

        auto conv3 = Symbol("convolution")
                .SetInput("data", pool2)
                .SetParam("filter", ShapeN(3, 3))
                .SetParam("padding", ShapeN(1, 1))
                .SetParam("stride", ShapeN(1, 1))
                .SetParam("dilate", ShapeN(1, 1))
                .SetParam("filter_num", 384)
                .SetParam("with_bias", true)
                .SetParam("group", 1)
                .SetParam("activation_type", kRelu)
                .Build("conv3");


        auto conv4 = Symbol("convolution")
                .SetInput("conv3", conv3)
                .SetParam("filter", ShapeN(3, 3))
                .SetParam("padding", ShapeN(1, 1))
                .SetParam("stride", ShapeN(1, 1))
                .SetParam("dilate", ShapeN(1, 1))
                .SetParam("filter_num", 384)
                .SetParam("with_bias", true)
                .SetParam("group", 1)
                .SetParam("activation_type", kRelu)
                .Build("conv4");


        auto conv5 = Symbol("convolution")
                .SetInput("conv4", conv4)
                .SetParam("filter", ShapeN(3, 3))
                .SetParam("padding", ShapeN(1, 1))
                .SetParam("stride", ShapeN(1, 1))
                .SetParam("dilate", ShapeN(1, 1))
                .SetParam("filter_num", 256)
                .SetParam("group", 1)
                .SetParam("with_bias", true)
                .SetParam("activation_type", kRelu)
                .Build("conv5");


        auto pool5 = Symbol("pooling")
                .SetInput("conv5", conv5)
                .SetParam("filter", ShapeN(3, 3))
                .SetParam("stride", ShapeN(2, 2))
                .SetParam("type", kMax)
                .Build("pool5");

        auto flatten = Symbol("flatten")
                .SetInput("pool5", pool5)
                .Build("flatten");

        auto fc1 = Symbol("fullyconnected")
                .SetInput("flatten", flatten)
                .SetParam("hide_num", 4096)
                .SetParam("with_bias", true)
                .SetParam("activation_type", kRelu)
                .Build("fc1");


        auto drop = Symbol("dropout")
                .SetInput("fc1", fc1)
                .SetParam("rate", 0.5f)
                .Build("drop");

        auto fc2 = Symbol("fullyconnected")
                .SetInput("drop", drop)
                .SetParam("hide_num", 4096)
                .SetParam("with_bias", true)
                .SetParam("activation_type", kRelu)
                .Build("fc2");

        auto drop2 = Symbol("dropout")
                .SetInput("fc2", fc2)
                .SetParam("rate", 0.5f)
                .Build("drop2");

        auto fc3 = Symbol("fullyconnected")
                .SetInput("drop2", drop2)
                .SetParam("hide_num", class_num)
                .SetParam("with_bias", true)
                .Build("fc3");

        auto softmax = Symbol("output")
                .SetInput("fc3", fc3)
                .SetParam("type", kSoftmax)
                .Build("softmax");

        return softmax;
    }

    Symbol MnistFullyConnected(const Symbol &input, int hideNum, int classNum) {

        auto y1 = Symbol("fullConnected")
                .SetInput("data", input)
                .SetParam("hide_num", hideNum)
                .SetParam("with_bias", true)
                .SetParam("activation_type", kRelu)
                .Build("y1");

        auto y2 = Symbol("fullConnected")
                .SetInput("y1", y1)
                .SetParam("hide_num", hideNum/2)
                .SetParam("with_bias", true)
                .Build("y2");

        auto drop = Symbol("dropout")
                .SetInput("y2", y2)
                .SetParam("rate", 0.2f)
                .Build("drop");

        auto y3 = Symbol("fullConnected")
                .SetInput("drop", drop)
                .SetParam("hide_num", classNum)
                .SetParam("with_bias", true)
                .Build("y3");

        auto out = Symbol("output")
                .SetInput("y3", y3)
                .SetParam("type", kSoftmax)
                .Build("out");

        return out;
    }

    Symbol MnistConvolution(const Symbol &input, int hideNum, int classNum) {

        auto conv1 = Symbol("convolution")
                .SetInput("data", input)
                .SetParam("filter", ShapeN(5, 5))
                .SetParam("with_bias", true)
                .SetParam("filter_num", 32)
                .SetParam("padding", ShapeN(0, 0))
                .SetParam("stride", ShapeN(1, 1))
                .SetParam("dilate", ShapeN(1, 1))
                .SetParam("group", 1)
                .SetParam("activation_type", kRelu)
                .Build("conv1");


        auto pool1 = Symbol("pooling")
                .SetInput("conv1", conv1)
                .SetParam("filter", ShapeN(2, 2))
                .SetParam("stride", ShapeN(1, 1))
                .SetParam("type", PoolType::kMax)
                .Build("pool1");


        auto conv2 = Symbol("convolution")
                .SetInput("pool1", pool1)
                .SetParam("filter", ShapeN(3, 3))
                .SetParam("with_bias", true)
                .SetParam("filter_num", 64)
                .SetParam("padding", ShapeN(0, 0))
                .SetParam("stride", ShapeN(1, 1))
                .SetParam("dilate", ShapeN(1, 1))
                .SetParam("group", 1)
                .SetParam("activation_type", kRelu)
                .Build("conv2");

        auto pool2 = Symbol("pooling")
                .SetInput("conv2", conv2)
                .SetParam("filter", ShapeN(2, 2))
                .SetParam("stride", ShapeN(1, 1))
                .SetParam("type", PoolType::kMax)
                .Build("pool2");

        auto conv3 = Symbol("convolution")
                .SetInput("pool2", pool2)
                .SetParam("filter", ShapeN(3, 3))
                .SetParam("with_bias", true)
                .SetParam("filter_num", 64)
                .SetParam("padding", ShapeN(0, 0))
                .SetParam("stride", ShapeN(1, 1))
                .SetParam("dilate", ShapeN(1, 1))
                .SetParam("group", 1)
                .SetParam("activation_type", kRelu)
                .Build("conv3");

        auto pool3 = Symbol("pooling")
                .SetInput("conv3", conv3)
                .SetParam("filter", ShapeN(2, 2))
                .SetParam("stride", ShapeN(1, 1))
                .SetParam("type", PoolType::kMax)
                .Build("pool3");


        auto flatten = Symbol("flatten")
                .SetInput("pool3", pool3)
                .Build("flatten");


        auto fc1 = Symbol("fullConnected")
                .SetInput("flatten", flatten)
                .SetParam("hide_num", hideNum)
                .SetParam("with_bias", true)
                .SetParam("activation_type", kRelu)
                .Build("fc1");

        auto fc2 = Symbol("fullConnected")
                .SetInput("fc1", fc1)
                .SetParam("hide_num", classNum)
                .SetParam("with_bias", true)
                .Build("fc2");

        auto out = Symbol("output")
                .SetInput("fc2", fc2)
                .SetParam("type", kSoftmax)
                .Build("out");

        return out;
    }


    /**
     * Convolution Op
     * @param input 输入数据
     * @param filter Kernel
     * @param filter_num Kernel channel
     * @param padding
     * @param stride
     * @param dilate
     * @param with_bias
     * @param group
     * @param actType 激活方式
     * @param name
     * @return
     */
    Symbol Convolution(const Symbol &input, Shape &filter, int filter_num,
                       const Shape &padding = ShapeN(0, 0),
                       const Shape &stride = ShapeN(1, 1),
                       const Shape &dilate = ShapeN(1, 1),
                       const bool with_bias = true,
                       const int group = 1,
                       const ActType &actType = kRelu,
                       const std::string &name = "conv") {
        auto conv = Symbol("convolution")
                .SetInput("data", input)
                .SetParam("filter", filter)
                .SetParam("with_bias", with_bias)
                .SetParam("filter_num", filter_num)
                .SetParam("padding", padding)
                .SetParam("stride", stride)
                .SetParam("dilate", dilate)
                .SetParam("group", 1)
                .SetParam("activation_type", actType)
                .Build(name);
        return conv;
    }


    /**
     * FullyConnected Op
     * @param input 输入数据
     * @param hideNum
     * @param with_bias
     * @param actType 激活方式
     * @param name
     * @return
     */
    Symbol FullyConnected(const Symbol &input, int hideNum,
                          const bool with_bias = true,
                          const ActType &actType = kRelu,
                          const std::string &name = "fullConnected") {
        auto fullConnected = Symbol("fullConnected")
                .SetInput("data", input)
                .SetParam("hide_num", hideNum)
                .SetParam("with_bias", with_bias)
                .SetParam("activation_type", actType)
                .Build(name);
        return fullConnected;
    }


    /**
     * Output Op
     * @param input
     * @param mode
     * @param name
     * @return
     */
    Symbol Output(const Symbol &input,
                  const OutputMode &mode = kSoftmax,
                  const std::string &name = "output") {
        auto out = Symbol("output")
                .SetInput("data", input)
                .SetParam("type", mode)
                .Build(name);
        return out;
    }


    /**
     * dropout Op
     * @param input
     * @param rate
     * @param name
     * @return
     */
    Symbol Dropout(const Symbol &input,
                   const float rate = 0.5f,
                   const std::string &name = "drop") {
        auto drop = Symbol("dropout")
                .SetInput("data", input)
                .SetParam("rate", rate)
                .Build(name);
        return drop;
    }



    /**
     * Pooling Op
     * @param input
     * @param filter
     * @param stride
     * @param padding
     * @param dilate
     * @param type
     * @param name
     * @return
     */
    Symbol Pooling(const Symbol &input,
                   const Shape &filter,
                   const Shape &stride=ShapeN(1, 1),
                   const Shape &padding=ShapeN(0, 0),
                   const Shape &dilate = ShapeN(1, 1),
                   const PoolType &type=kMax,
                   const std::string &name = "pooling") {
        auto pool = Symbol("pooling")
                .SetInput("data", input)
                .SetParam("filter", filter)
                .SetParam("padding", padding)
                .SetParam("stride", stride)
                .SetParam("dilate", dilate)
                .SetParam("type", type)
                .Build(name);
        return pool;
    }


    /**
     * Flatten Op
     * @param input
     * @param shape
     * @param name
     * @return
     */
    Symbol Flatten(const Symbol &input,
                   const Shape *shape = nullptr,
                   const std::string &name = "flatten") {
        auto flatten = Symbol("flatten")
                .SetInput("data", input);
        if (shape != nullptr) {
            flatten.SetParam("shape", *shape);
        }
        flatten.Build("name");
        return flatten;
    }


    /**
     * Loss Op
     * @param input
     * @param label
     * @param mode
     * @param name
     * @return
     */
    Symbol Loss(const Symbol &input, const Symbol &label,
                const LossMode &mode = kCrossEntropy,
                const std::string &name = "loss") {
        auto loss = Symbol("loss")
                .SetInput("data", input)
                .SetInput("label", label)
                .SetParam("type", mode)
                .Build(name);
        return loss;
    }

    /**
     * Concat Op
     * @param inputs 输入数据
     * @param axis 维度
     * @param name
     * @return
     */
    Symbol Concat(const std::vector<Symbol> &inputs,
                  const int axis =-1,
                  const std::string &name = "concat") {
        auto concat = Symbol("concat")
                .SetParam("axis", axis);
        for (auto &input : inputs) {
            concat.SetInput("data", input);
        }
        concat.Build(name);
        return concat;
    }


    /**
     * Accuracy Op
     * @param input
     * @param label
     * @param name
     * @return
     */
    Symbol Accuracy(const Symbol &input, const Symbol &label,
                    const std::string &name = "accuracy") {
        auto acc = Symbol("accuracy")
                .SetInput("data", input)
                .SetInput("label", label)
                .Build(name);
        return acc;
    }



    /**
     * Activation Op
     * @param input
     * @param type
     * @param name
     * @return
     */
    Symbol Activation(const Symbol &input,
                      const ActType &type=kRelu,
                      const std::string &name = "activation") {
        auto activation = Symbol("activation")
                .SetInput("data", input)
                .SetParam("type", type)
                .Build(name);
        return activation;
    }


}


#endif //MATRIX_ALEXNET_H
