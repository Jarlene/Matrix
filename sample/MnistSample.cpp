//
// Created by Jarlene on 2017/7/18.
//

#include <matrix/include/api/Symbol.h>
#include <matrix/include/api/VariableSymbol.h>
#include <matrix/include/api/PlaceHolderSymbol.h>
#include <matrix/include/models/AlexNet.h>
#include <matrix/include/executor/Executor.h>
#include "include/DataSet.h"

using namespace matrix;

const string trainImagePath = "../../data/train-images-idx3-ubyte";
const string trainLabelPath = "../../data/train-labels-idx1-ubyte";


Symbol LogisticRegression(Symbol input, int hideNum, int classNum) {

    auto y1 = Symbol("fullConnected")
            .SetInput("data", input)
            .SetParam("weight", ShapeN(784, hideNum))
            .SetParam("bias", true)
            .Build();

    auto act1 = Symbol("activation")
            .SetInput("y1", y1)
            .SetParam("type", kSigmoid)
            .Build();

    auto y2 = Symbol("fullConnected")
            .SetInput("act1", act1)
            .SetParam("weight", ShapeN(hideNum, classNum))
            .SetParam("bias", true)
            .Build();

    auto out = Symbol("output")
            .SetInput("y2", y2)
            .SetParam("type", kSoftmax)
            .Build();

    return out;
}


Symbol Connvolution(Symbol &input, int batchSize, int classNum) {

    auto conv1 = Symbol("convolution")
            .SetInput("data", input)
            .SetParam("filter", ShapeN(3, 3))
            .SetParam("bias", true)
            .SetParam("padding", ShapeN(0,0))
            .SetParam("stride", ShapeN(1,1))
            .SetParam("dilate", ShapeN(1,1))
            .SetParam("filter_num", 8)
            .SetParam("group", 1)
            .Build();

    auto act = Symbol("activation")
            .SetInput("conv1", conv1)
            .SetParam("type", ActType::kRelu)
            .Build();

    auto pool1 = Symbol("pooling")
            .SetInput("act", act)
            .SetParam("filter", ShapeN(3,3))
            .SetParam("type", PoolType::kMax)
            .Build();


    auto conv2 = Symbol("convolution")
            .SetInput("pool1", pool1)
            .SetParam("filter",  ShapeN(2, 2))
            .SetParam("bias", true)
            .SetParam("padding", ShapeN(0,0))
            .SetParam("stride", ShapeN(1,1))
            .SetParam("dilate", ShapeN(1,1))
            .SetParam("filter_num", 16)
            .SetParam("group", 1)
            .Build();

    auto act2 = Symbol("activation")
            .SetInput("conv2", conv2)
            .SetParam("type", ActType::kRelu)
            .Build();

    auto pool2 = Symbol("pooling")
            .SetInput("act2", act2)
            .SetParam("filter", ShapeN(2,2))
            .SetParam("type", PoolType::kMax)
            .Build();

    auto flatten = Symbol("flatten")
            .SetInput("pool2", pool2)
            .Build();

    auto fully = Symbol("fullConnected")
            .SetInput("flatten", flatten)
            .SetParam("weight", ShapeN(864, classNum))
            .SetParam("bias", true)
            .Build();

    auto out = Symbol("output")
            .SetInput("fully", fully)
            .SetParam("type", kSoftmax)
            .Build();

    return fully;
}



int main() {
    const int batchSize = 100;
    const int epochSize = 10;
    const int classNum = 10;
    const int hideNum = 128;

    Shape imageShape = ShapeN(batchSize, 1, 28, 28);
    Shape labelShape = ShapeN(batchSize);
    auto image = PlaceHolderSymbol::Create("image", imageShape);
    auto label = PlaceHolderSymbol::Create("label", labelShape);

    float* imageData = static_cast<float *>(malloc(sizeof(float) * imageShape.Size()));
    float* labelData = static_cast<float *>(malloc(sizeof(float) * labelShape.Size()));


    auto convolution = Connvolution(image, batchSize, classNum);

    auto logistic = LogisticRegression(image, hideNum, classNum);

    auto loss = Symbol("loss")
            .SetInput("logistic", convolution)
            .SetInput("y", label)
            .SetParam("type", kCrossEntropy)
            .Build();

    auto prediction = Symbol("prediction")
            .SetInput("logistic", convolution)
            .SetInput("y", label)
            .Build();

    Context context;
    context.type = kFloat;
    context.phase = TRAIN;
    context.mode = kCpu;
    DataSet trainSet(trainImagePath, trainLabelPath);
    auto executor = std::make_shared<Executor>(loss, context);

    for (int i = 0; i < epochSize; ++i) {
        trainSet.GetBatchData(batchSize, imageData, labelData);
        input.Fill(imageData);
        label.Fill(labelData);
        executor->runSync();
    }

    free(imageData);
    free(labelData);

    return 0;
}