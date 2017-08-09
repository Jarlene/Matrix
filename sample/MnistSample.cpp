//
// Created by Jarlene on 2017/7/18.
//

#include <matrix/include/api/Symbol.h>
#include <matrix/include/api/VariableSymbol.h>
#include <matrix/include/api/PlaceHolderSymbol.h>
#include <matrix/include/api/ConstantSymbol.h>

using namespace matrix;



Symbol LogisticRegression(Symbol input, int batchSize, int hideNum) {
    auto w1 = VariableSymbol::Create("w1", ShapeN(784, hideNum));
    auto b1 = VariableSymbol::Create("b1", ShapeN(batchSize));
    auto y = input * w1 + b1 ;
    auto w2 = VariableSymbol::Create("w2", ShapeN(hideNum, 10));
    auto b2 = VariableSymbol::Create("b2", ShapeN(batchSize));
    auto out = y * w2 + b2;
    return out;
}


Symbol Connvolution(Symbol input, int batchSize) {
    int a = 2;
    auto symbol = ConstantSymbol::Create<int>("constant", a);
    return Symbol("");
}



int main() {
    int batchSize = 100;
    auto input = PlaceHolderSymbol::Create("x", ShapeN(batchSize, 784));
    auto label = PlaceHolderSymbol::Create("label", ShapeN(batchSize));
    auto symbol = LogisticRegression(input, batchSize, 10);

    auto loss = Symbol("crossEntropy")
            .SetInput("logistic", symbol)
            .SetInput("y", label)
            .Build();

    auto prediction = Symbol("prediction")
            .SetInput("logistic", symbol)
            .SetInput("y", label)
            .Build();


    return 0;
}