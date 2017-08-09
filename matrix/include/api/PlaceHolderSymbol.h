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
        static PlaceHolderSymbol Create(const std::string &name, const Shape &shape, const MatrixType &type = kFloat);


        template <class T>
        void Fill(T *ptr) {
            this->nodePtr->data_ = static_cast<void*>(ptr);
        }

    private:
        PlaceHolderSymbol();
        explicit PlaceHolderSymbol(const std::string &name);
    };




}

#endif //MATRIX_PLACEHOLDERSYMBOL_H
