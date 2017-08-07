//
// Created by Jarlene on 2017/7/21.
//

#include "matrix/include/api/Symbol.h"


namespace matrix {


    Symbol::Symbol(const Symbol &symbol) {
        this->nodePtr = symbol.nodePtr;
    }

    Symbol::Symbol(const std::string &name) {
        nodePtr = Node::Create();
        nodePtr->opName = name;
    }

    Symbol &Symbol::SetInput(const std::string &name, const Symbol &symbol) {
        this->nodePtr->inputs.push_back(symbol.nodePtr);
        symbol.nodePtr->outputs.push_back(std::weak_ptr<Node>(this->nodePtr));
        return *this;
    }

    Symbol &Symbol::SetParam(const std::string &name, const Any &value) {
        this->nodePtr->params[name] = value;
        return *this;
    }

    Symbol &Symbol::Build() {
        this->nodePtr->Build();
        return *this;
    }

    Symbol &Symbol::operator=(const Symbol &symbol) {
        this->nodePtr = symbol.nodePtr;
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

