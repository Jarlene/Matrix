//
// Created by Jarlene on 2017/7/21.
//

#ifndef MATRIX_PLACEHOLDERSYMBOL_H
#define MATRIX_PLACEHOLDERSYMBOL_H

#include "Symbol.h"
#include "matrix/include/base/Shape.h"

namespace matrix {
    class PlaceHolderSymbol : public Symbol {
    public:
        static Symbol Create(const std::string &name, const Shape &shape, const MatrixType &type = kFloat);


        template <class T>
        void Fill(T *ptr) {

        }

    };




}

#endif //MATRIX_PLACEHOLDERSYMBOL_H
