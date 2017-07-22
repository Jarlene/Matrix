//
// Created by Jarlene on 2017/7/21.
//

#ifndef MATRIX_VARIABLESYMBOL_H
#define MATRIX_VARIABLESYMBOL_H

#include "Symbol.h"
#include "matrix/include/base/Shape.h"

namespace matrix {

    class VariableSymbol : public Symbol {
    public:
        static Symbol Create(const std::string &name, const Shape &shape);
    };


    Symbol matrix::VariableSymbol::Create(const std::string &name, const matrix::Shape &shape) {
        return matrix::Symbol();
    }
}

#endif //MATRIX_VARIABLESYMBOL_H
