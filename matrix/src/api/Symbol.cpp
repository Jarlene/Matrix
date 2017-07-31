//
// Created by Jarlene on 2017/7/21.
//

#include "matrix/include/api/Symbol.h"


namespace matrix {


    Symbol::Symbol(const Symbol &symbol) {

    }

    Symbol::Symbol(const std::string &name) {

    }

    Symbol &Symbol::SetInput(const std::string &name, const Symbol &symbol) {
        return *this;
    }

    Symbol &Symbol::SetParam(const std::string &name, const Any &value) {
        return *this;
    }

    Symbol &Symbol::Build() {
        return *this;
    }

    Symbol &Symbol::operator=(const Symbol &symbol) {
        return *this;
    }

    Symbol Symbol::operator+(const Symbol &symbol) {
        auto s = Symbol("add")
                .SetInput("first", *this)
                .SetInput("third", symbol)
                .Build();
        return s;
    }

    Symbol Symbol::operator-(const Symbol &symbol) {
        auto s = Symbol("sub")
                .SetInput("first", *this)
                .SetInput("third", symbol)
                .Build();
        return s;
    }

    Symbol Symbol::operator*(const Symbol &symbol) {
        auto s = Symbol("mul")
                .SetInput("first", *this)
                .SetInput("third", symbol)
                .Build();
        return s;
    }

    Symbol Symbol::operator/(const Symbol &symbol) {
        auto s = Symbol("div")
                .SetInput("first", *this)
                .SetInput("third", symbol)
                .Build();
        return s;
    }

}

