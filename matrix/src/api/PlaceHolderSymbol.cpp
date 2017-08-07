//
// Created by Jarlene on 2017/7/31.
//

#include "matrix/include/api/PlaceHolderSymbol.h"

namespace matrix {

    Symbol PlaceHolderSymbol::Create(const std::string &name, const Shape &shape, const MatrixType &type) {
        auto symbol = Symbol("placeHolder")
                .SetParam("shape", shape)
                .Build();
        return symbol;
    }


}
