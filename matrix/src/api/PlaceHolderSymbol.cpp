//
// Created by Jarlene on 2017/7/31.
//

#include "matrix/include/api/PlaceHolderSymbol.h"

namespace matrix {

    PlaceHolderSymbol PlaceHolderSymbol::Create(const std::string &name, const Shape &shape, const MatrixType &type) {
        auto symbol = PlaceHolderSymbol("placeHolder");
        symbol.nodePtr->outputShapes = shape;
        symbol.nodePtr->context.type = type;
        symbol.nodePtr->isPlaceHolder = true;
        symbol.Build(name);
        return symbol;
    }

    PlaceHolderSymbol::PlaceHolderSymbol() {

    }

    PlaceHolderSymbol::PlaceHolderSymbol(const std::string &name) : Symbol(name) {

    }


}
