//
// Created by Jarlene on 2017/7/18.
//

#include <matrix/include/api/Symbol.h>
#include <matrix/include/api/VariableSymbol.h>
#include <matrix/include/api/PlaceHolderSymbol.h>
#include <matrix/include/models/AlexNet.h>
#include <matrix/include/executor/Executor.h>

using namespace matrix;



Symbol LogisticRegression(Symbol input, int batchSize, int hideNum) {
    auto w1 = VariableSymbol::Create("w1", ShapeN(784, hideNum));
    auto b1 = VariableSymbol::Create("b1", ShapeN(batchSize));

    auto y1 = Symbol("fullConnected")
            .SetInput("x", input)
            .SetInput("w1", w1)
            .SetInput("b1", b1);

    auto act1 = Symbol("activation")
            .SetInput("y1", y1)
            .SetParam("type", kSigmoid);

    auto w2 = VariableSymbol::Create("w2", ShapeN(hideNum, 10));
    auto b2 = VariableSymbol::Create("b2", ShapeN(batchSize));

    auto y2 = Symbol("fullConnected")
            .SetInput("act1", act1)
            .SetInput("w2", w2)
            .SetInput("b2", b2);

    auto out = Symbol("output")
            .SetInput("y2", y2)
            .SetParam("type", kSoftmax);

    return out;
}


Symbol Connvolution(Symbol &input, Symbol &label, int batchSize) {

    auto symbol = AlexSymbol(input, label);
    return symbol;
}



int main() {
    int batchSize = 100;
    int epochSize = 200;
    auto input = PlaceHolderSymbol::Create("x", ShapeN(batchSize, 784));
    auto label = PlaceHolderSymbol::Create("label", ShapeN(batchSize));
    auto symbol = LogisticRegression(input, batchSize, 128);

    auto loss = Symbol("loss")
            .SetInput("logistic", symbol)
            .SetInput("y", label)
            .SetParam("type", kCrossEntropy);

    auto prediction = Symbol("prediction")
            .SetInput("logistic", symbol)
            .SetInput("y", label);

    Context context;
    context.type = kFloat;
    context.phase = TRAIN;
    context.mode = kCpu;

    auto executor = std::make_shared<Executor>(loss, context);

    for (int i = 0; i < epochSize; ++i) {
        executor->runSync();
    }

    return 0;
}