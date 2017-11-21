//
// Created by Jarlene on 2017/7/18.
//
#include <matrix/include/utils/Logger.h>
#include <matrix/include/api/Symbol.h>
#include <matrix/include/optimizer/SGDOptimizer.h>
#include <matrix/include/api/PlaceHolderSymbol.h>
#include <matrix/include/executor/Executor.h>
#include "include/DataSet.h"

using namespace matrix;

const string trainImagePath = "../../data/train-images-idx3-ubyte";
const string trainLabelPath = "../../data/train-labels-idx1-ubyte";
const string testImagePath = "../../t10k-images-idx3-ubyte";
const string testLabelPath = "../../t10k-labels-idx1-ubyte";


Symbol LogisticRegression(Symbol input, int hideNum, int classNum) {

    auto y1 = Symbol("fullConnected")
            .SetInput("data", input)
            .SetParam("hide_num", hideNum)
            .SetParam("with_bias", true)
            .Build("f1");

    auto act1 = Symbol("activation")
            .SetInput("y1", y1)
            .SetParam("type", kSigmoid)
            .Build("act1");

    auto y2 = Symbol("fullConnected")
            .SetInput("act1", act1)
            .SetParam("hide_num", classNum)
            .SetParam("with_bias", true)
            .Build("y2");

    auto out = Symbol("output")
            .SetInput("y2", y2)
            .SetParam("type", kSoftmax)
            .Build("out");

    return out;
}


Symbol Connvolution(Symbol &input, int batchSize, int classNum) {

    auto conv1 = Symbol("convolution")
            .SetInput("data", input)
            .SetParam("filter", ShapeN(3, 3))
            .SetParam("with_bias", true)
            .SetParam("padding", ShapeN(0,0))
            .SetParam("stride", ShapeN(1,1))
            .SetParam("dilate", ShapeN(1,1))
            .SetParam("filter_num", 16)
            .SetParam("group", 1)
            .Build("conv1");

    auto act = Symbol("activation")
            .SetInput("conv1", conv1)
            .SetParam("type", ActType::kRelu)
            .Build("act");

    auto pool1 = Symbol("pooling")
            .SetInput("act", act)
            .SetParam("filter", ShapeN(2,2))
            .SetParam("type", PoolType::kMax)
            .Build("pool1");


    auto conv2 = Symbol("convolution")
            .SetInput("pool1", pool1)
            .SetParam("filter",  ShapeN(2, 2))
            .SetParam("with_bias", true)
            .SetParam("padding", ShapeN(0,0))
            .SetParam("stride", ShapeN(1,1))
            .SetParam("dilate", ShapeN(1,1))
            .SetParam("filter_num", 32)
            .SetParam("group", 1)
            .Build("conv2");

    auto act2 = Symbol("activation")
            .SetInput("conv2", conv2)
            .SetParam("type", ActType::kRelu)
            .Build("act2");

    auto pool2 = Symbol("pooling")
            .SetInput("act2", act2)
            .SetParam("filter", ShapeN(2,2))
            .SetParam("type", PoolType::kMax)
            .Build("pool2");

    auto flatten = Symbol("flatten")
            .SetInput("pool2", pool2)
            .Build("flatten");

    auto fc = Symbol("fullConnected")
            .SetInput("flatten", flatten)
            .SetParam("hide_num", 64)
            .SetParam("with_bias", true)
            .Build("fc1");

    auto fc2 = Symbol("fullConnected")
            .SetInput("flatten", fc)
            .SetParam("hide_num", classNum)
            .SetParam("with_bias", true)
            .Build("fc2");

    auto out = Symbol("output")
            .SetInput("fully", fc2)
            .SetParam("type", kSoftmax)
            .Build("out");

    return out;
}



int main() {
    const int batchSize = 300;
    const int epochSize = 10000;
    const int classNum = 10;
    const int hideNum = 128;

    Shape imageShape = ShapeN(batchSize,  784);
    Shape labelShape = ShapeN(batchSize);
    auto image = PlaceHolderSymbol::Create("image", imageShape);
    auto label = PlaceHolderSymbol::Create("label", labelShape);

    float* imageData = static_cast<float *>(malloc(sizeof(float) * imageShape.Size()));
    float* labelData = static_cast<float *>(malloc(sizeof(float) * labelShape.Size()));


//    auto logistic = Connvolution(image, batchSize, classNum);

    auto logistic = LogisticRegression(image, hideNum, classNum);

    auto loss = Symbol("loss")
            .SetInput("logistic", logistic)
            .SetInput("y", label)
            .SetParam("type", kCrossEntropy)
            .Build("loss");

    auto acc = Symbol("accuracy")
            .SetInput("logistic", logistic)
            .SetInput("y", label)
            .Build("acc");


    Context context;
    context.type = kFloat;
    context.phase = TRAIN;
    context.mode = kCpu;
    auto opt = new SGDOptimizer(0.01f);
    DataSet trainSet(trainImagePath, trainLabelPath);
//    DataSet testSet(testImagePath, testLabelPath);
    auto executor = std::make_shared<Executor>(loss, context, opt);
    trainSet.GetBatchData(batchSize, imageData, labelData);
    image.Fill(imageData);
    label.Fill(labelData);
    for (int i = 0; i < epochSize; ++i) {
        executor->train(&acc);
        loss.PrintMatrix();
        acc.PrintMatrix();
        trainSet.Reset();
    }

    free(imageData);
    free(labelData);

    return 0;
}