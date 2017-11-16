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
        symbol.SetParam("isTrain", true);
        symbol.nodePtr->isVariable = true;
        symbol.Build(name);
        return symbol;
    }

    VariableSymbol::VariableSymbol() {

    }

    VariableSymbol::VariableSymbol(const std::string &name) : Symbol(name) {

    }

    VariableSymbol VariableSymbol::Create(const std::string &name, const MatrixType &type) {
        auto symbol = VariableSymbol("variable");
        symbol.SetParam("isTrain", true);
        symbol.nodePtr->isVariable = true;
        symbol.nodePtr->outputShapes = ShapeN(0,0);
        symbol.nodePtr->nodeName = name;
        symbol.nodePtr->context.type = type;
        symbol.Build(name);
        return symbol;
    }

}