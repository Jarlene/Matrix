//
// Created by Jarlene on 2017/7/21.
//

#ifndef MATRIX_CONSTANTSYMBOL_H
#define MATRIX_CONSTANTSYMBOL_H

#include "Symbol.h"
#include "matrix/include/base/Shape.h"

namespace matrix {

    class ConstantSymbol : public Symbol {
    public:
        template <class T>
        static Symbol Create(const std::string &name, const T &value);

        template <class T>
        static Symbol Create(const std::string &name,  T *values, Shape &shape);


        template <class T>
        static Symbol Create(const std::string &name, const std::vector<T> &values, Shape &shape);

    private:
        ConstantSymbol();
        explicit ConstantSymbol(const std::string &name);
    };




    template<class T>
    Symbol ConstantSymbol::Create(const std::string &name, const T &value) {
        auto symbol = ConstantSymbol(name);
        T* s = const_cast<T*>(&value);
        symbol.nodePtr->data_ = s;
        symbol.Build();
        return symbol;
    }

    template<class T>
    Symbol ConstantSymbol::Create(const std::string &name,  T *values, Shape &shape) {
        auto symbol = ConstantSymbol();
        symbol.nodePtr->data_ = values;
        symbol.nodePtr->nodeName = name;
        symbol.nodePtr->outputShapes = shape;
        symbol.Build();
        return symbol;
    }

    template<class T>
    Symbol ConstantSymbol::Create(const std::string &name, const std::vector<T> &values, Shape &shape) {
        auto symbol = ConstantSymbol(name);
        symbol.nodePtr->nodeName = name;
        symbol.nodePtr->data_ = values.data();
        symbol.nodePtr->outputShapes = shape;
        symbol.Build();
        return symbol;
    }

    ConstantSymbol::ConstantSymbol() : Symbol("") {

    }

    ConstantSymbol::ConstantSymbol(const std::string &name) : Symbol("") {

    }
}

#endif //MATRIX_CONSTANTSYMBOL_H
