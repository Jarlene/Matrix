//
// Created by Jarlene on 2017/7/31.
//


#include "matrix/include/api/VariableSymbol.h"

namespace matrix {

    Symbol VariableSymbol::Create(const std::string &name, const Shape &shape,  const MatrixType &type) {
        auto symbol = Symbol("variable")
                .SetParam("shape", shape)
                .Build();
        return symbol;
    }

}