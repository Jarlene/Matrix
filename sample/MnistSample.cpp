//
// Created by Jarlene on 2017/7/18.
//

#include <matrix/include/api/Symbol.h>
#include <matrix/include/api/VariableSymbol.h>
#include <matrix/include/api/PlaceHolderSymbol.h>
#include <matrix/include/models/AlexNet.h>
using namespace matrix;



Symbol LogisticRegression(Symbol input, int batchSize, int hideNum) {
    auto w1 = VariableSymbol::Create("w1", ShapeN(784, hideNum));
    auto b1 = VariableSymbol::Create("b1", ShapeN(batchSize));

    auto y1 = Symbol("fullConnected")
            .SetInput("x", input)
            .SetInput("w", w1)
            .SetInput("b", b1);

    auto act1 = Symbol("activation")
            .SetInput("y1", y1)
            .SetParam("type", kSigmoid);

    auto w2 = VariableSymbol::Create("w2", ShapeN(hideNum, 10));
    auto b2 = VariableSymbol::Create("b2", ShapeN(batchSize));

    auto y2 = Symbol("fullConnected")
            .SetInput("x", act1)
            .SetInput("w", w2)
            .SetInput("b", b2);

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


    return 0;
}