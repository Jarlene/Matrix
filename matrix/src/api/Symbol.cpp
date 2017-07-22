//
// Created by Jarlene on 2017/7/21.
//

#include "matrix/include/api/Symbol.h"


namespace matrix {
    Symbol::Symbol() {

    }

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

    }

    Symbol Symbol::operator+(const Symbol &symbol) {

    }

    Symbol Symbol::operator-(const Symbol &symbol) {

    }

    Symbol Symbol::operator*(const Symbol &symbol) {

    }

    Symbol Symbol::operator/(const Symbol &symbol) {

    }

}

