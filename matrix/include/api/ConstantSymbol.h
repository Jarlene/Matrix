//
// Created by 郑珊 on 2017/7/21.
//

#ifndef MATRIX_CONSTANTSYMBOL_H
#define MATRIX_CONSTANTSYMBOL_H

#include <vector>
#include "Symbol.h"
#include "matrix/include/base/Shape.h"

namespace matrix {

    class ConstantSymbol : public Symbol {
    public:
        template <class T>
        static ConstantSymbol Create(const std::string &name, const T &value);

        template <class T>
        static ConstantSymbol Create(std::string &name, T *values, Shape &shape);


        template <class T>
        static ConstantSymbol Create(std::string &name, std::vector<T> &values, Shape &shape);
    };




    template<class T>
    ConstantSymbol ConstantSymbol::Create(const std::string &name, const T &value) {
        return ConstantSymbol();
    }

    template<class T>
    ConstantSymbol ConstantSymbol::Create(std::string &name, T *values, Shape &shape) {
        return ConstantSymbol();
    }

    template<class T>
    ConstantSymbol ConstantSymbol::Create(std::string &name, std::vector<T> &values, Shape &shape) {
        return ConstantSymbol();
    }
}

#endif //MATRIX_CONSTANTSYMBOL_H
