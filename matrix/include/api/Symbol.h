//
// Created by 郑珊 on 2017/7/18.
//

#ifndef MATRIX_SYMBOL_H
#define MATRIX_SYMBOL_H

#include <string>
#include "matrix/include/utils/Any.h"


namespace matrix {
    class Symbol {
    public:
        Symbol();

        Symbol(const Symbol &symbol);

        explicit Symbol(const std::string &name);

        Symbol &SetInput(const std::string &name, const Symbol &symbol);

        Symbol &SetParam(const std::string &name, const Any &value);

        template<class T>
        Symbol &SetParam(const std::string &name, const T &value) {
            return SetParam(name, Any(value));
        }

        Symbol &Build();

        Symbol &operator=(const Symbol &symbol);

        Symbol operator+(const Symbol &symbol);

        Symbol operator-(const Symbol &symbol);

        Symbol operator*(const Symbol &symbol);

        Symbol operator/(const Symbol &symbol);

    };
}



#endif //MATRIX_SYMBOL_H
