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


Symbol LogisticRegression(Symbol input, int batchSize, int hideNum) {
    auto w1 = VariableSymbol::Create("w1", ShapeN(784, hideNum));
    auto b1 = VariableSymbol::Create("b1", ShapeN(batchSize));

    auto y1 = Symbol("fullConnected")
            .SetInput("x", input)
            .SetInput("w1", w1)
            .SetInput("b1", b1)
            .Build();

    auto act1 = Symbol("activation")
            .SetInput("y1", y1)
            .SetParam("type", kSigmoid)
            .Build();

    auto w2 = VariableSymbol::Create("w2", ShapeN(hideNum, 10));
    auto b2 = VariableSymbol::Create("b2", ShapeN(batchSize));

    auto y2 = Symbol("fullConnected")
            .SetInput("act1", act1)
            .SetInput("w2", w2)
            .SetInput("b2", b2)
            .Build();

    auto out = Symbol("output")
            .SetInput("y2", y2)
            .SetParam("type", kSoftmax)
            .Build();

    return out;
}


Symbol Connvolution(Symbol &input, Symbol &label, int batchSize, int classNum) {
    auto w = VariableSymbol::Create("w1", ShapeN(3, 3));
    auto b = VariableSymbol::Create("b1", ShapeN(batchSize));

    auto conv1 = Symbol("convolution")
            .SetInput("data", input)
            .SetInput("kernel", w)
            .SetInput("bias", b)
            .SetParam("padding", ShapeN(0,0))
            .SetParam("stride", ShapeN(1,1))
            .SetParam("dilate", ShapeN(1,1))
            .SetParam("filter_num", 8)
            .Build();

    auto act = Symbol("activation")
            .SetInput("data", conv1)
            .SetParam("type", ActType::kRelu)
            .Build();

    auto pool1 = Symbol("pooling")
            .SetInput("data", act)
            .SetParam("filter", ShapeN(3,3))
            .SetParam("type", PoolType::kMax)
            .Build();

    auto w2 = VariableSymbol::Create("w2", ShapeN(2, 2));
    auto b2 = VariableSymbol::Create("b2", ShapeN(batchSize));

    auto conv2 = Symbol("convolution")
            .SetInput("data", pool1)
            .SetInput("kernel", w2)
            .SetInput("bias", b)
            .SetParam("padding", ShapeN(0,0))
            .SetParam("stride", ShapeN(1,1))
            .SetParam("dilate", ShapeN(1,1))
            .SetParam("filter_num", 16)
            .Build();

    auto act2 = Symbol("activation")
            .SetInput("data", conv2)
            .SetParam("type", ActType::kRelu)
            .Build();

    auto pool2 = Symbol("pooling")
            .SetInput("data", act2)
            .SetParam("filter", ShapeN(2,2))
            .SetParam("type", PoolType::kMax)
            .Build();

    auto flatten = Symbol("flatten")
            .SetInput("data", pool2)
            .Build();

    auto w3 = VariableSymbol::Create("w3", ShapeN(7744, classNum));
    auto b3 = VariableSymbol::Create("b3", ShapeN(batchSize));

    auto fully = Symbol("fullConnected")
            .SetInput("data", flatten)
            .SetInput("weight", w3)
            .SetInput("bias", b3)
            .Build();
    return fully;
}



int main() {
    const int batchSize = 100;
    const int epochSize = 10;
    const int classNum = 10;

    Shape imageShape = ShapeN(batchSize, 1, 28, 28);
    Shape labelShape = ShapeN(batchSize);
    auto input = PlaceHolderSymbol::Create("x", imageShape);
    auto label = PlaceHolderSymbol::Create("label", labelShape);

    float* imageData = static_cast<float *>(malloc(sizeof(float) * imageShape.Size()));
    float* labelData = static_cast<float *>(malloc(sizeof(float) * labelShape.Size()));


    auto symbol = Connvolution(input, label, batchSize, classNum);

    auto loss = Symbol("loss")
            .SetInput("logistic", symbol)
            .SetInput("y", label)
            .SetParam("type", kCrossEntropy)
            .Build();

    auto prediction = Symbol("prediction")
            .SetInput("logistic", symbol)
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