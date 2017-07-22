//
// Created by 郑珊 on 2017/7/21.
//

#ifndef MATRIX_PLACEHOLDERSYMBOL_H
#define MATRIX_PLACEHOLDERSYMBOL_H

#include "Symbol.h"
#include "matrix/include/base/Shape.h"

namespace matrix {
    class PlaceHolderSymbol : public Symbol {
    public:
        static PlaceHolderSymbol Create(const std::string &name, const Shape &shape);
        void FeedData(void *ptr);
    };



    PlaceHolderSymbol PlaceHolderSymbol::Create(const std::string &name, const Shape &shape) {
        return PlaceHolderSymbol();
    }

    void PlaceHolderSymbol::FeedData(void *ptr) {

    }
}

#endif //MATRIX_PLACEHOLDERSYMBOL_H
