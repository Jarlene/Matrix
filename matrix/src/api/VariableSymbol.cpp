//
// Created by Jarlene on 2017/7/31.
//


#include "matrix/include/api/VariableSymbol.h"

namespace matrix {

    VariableSymbol VariableSymbol::Create(const std::string &name, const Shape &shape,  const MatrixType &type) {
        auto symbol = VariableSymbol("variable");
        symbol.nodePtr->outputShapes = shape;
        symbol.nodePtr->context.type = type;
        symbol.nodePtr->nodeName = name;
        symbol.nodePtr->isVariable = true;
        symbol.Build();
        return symbol;
    }

    VariableSymbol::VariableSymbol() {

    }

    VariableSymbol::VariableSymbol(const std::string &name) : Symbol(name) {

    }

}