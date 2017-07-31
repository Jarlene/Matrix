//
// Created by Jarlene on 2017/7/31.
//

#include "matrix/include/api/PlaceHolderSymbol.h"

namespace matrix {

    Symbol PlaceHolderSymbol::Create(const std::string &name, const Shape &shape) {
        auto symbol = Symbol("placeHolder")
                .SetParam("shape", shape)
                .Build();
        return symbol;
    }

    void PlaceHolderSymbol::FillData(void *ptr) {

    }


}
