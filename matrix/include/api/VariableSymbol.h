//
// Created by Jarlene on 2017/7/21.
//

#ifndef MATRIX_VARIABLESYMBOL_H
#define MATRIX_VARIABLESYMBOL_H

#include "Symbol.h"
#include "matrix/include/base/Shape.h"
#include "MatrixType.h"

namespace matrix {

    class VariableSymbol : public Symbol {
    public:
        static VariableSymbol Create(const std::string &name, const Shape &shape, const MatrixType &type = kFloat);

        template <class T>
        void Fill(T * ptr) {
            this->nodePtr->context.phase = PREDICTION;
            this->nodePtr->data_ = ptr;
        }

    private:
        VariableSymbol();
        explicit VariableSymbol(const std::string &name);
    };


}

#endif //MATRIX_VARIABLESYMBOL_H
