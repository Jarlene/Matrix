//
// Created by Jarlene on 2017/7/21.
//

#include "matrix/include/api/Symbol.h"
#include "matrix/include/utils/StringUtil.h"

namespace matrix {


    Symbol::Symbol(const Symbol &symbol) {
        this->nodePtr = symbol.nodePtr;
    }

    Symbol::Symbol(const std::string &name) {
        nodePtr = Node::Create();
        nodePtr->AddOpName(name);
    }

    Symbol &Symbol::SetInput(const std::string &name, const Symbol &symbol) {
        this->nodePtr->AddInput(symbol.nodePtr);
        this->nodePtr->On(symbol.nodePtr->context);
        return *this;
    }

    Symbol &Symbol::SetParam(const std::string &name, const Any &value) {
        this->nodePtr->AddParam(name, value);
        return *this;
    }

    Symbol &Symbol::Build(const std::string &symbol_name) {
        this->nodePtr->AddNodeName(symbol_name + toString(this->nodePtr->id_));
        this->nodePtr->Build();
        return *this;
    }

    Symbol &Symbol::On(const Context &c) {
        this->nodePtr->context = c;
        return *this;
    }

    Symbol &Symbol::operator=(const Symbol &symbol) {
        this->nodePtr = symbol.nodePtr;
        return *this;
    }

    Symbol Symbol::operator+(const Symbol &symbol) {
        auto s = Symbol("add")
                .SetInput("first", *this)
                .SetInput("second", symbol)
                .Build("add");
        return s;
    }

    Symbol Symbol::operator-(const Symbol &symbol) {
        auto s = Symbol("sub")
                .SetInput("first", *this)
                .SetInput("second", symbol)
                .Build("sub");
        return s;
    }

    Symbol Symbol::operator*(const Symbol &symbol) {
        auto s = Symbol("mul")
                .SetInput("first", *this)
                .SetInput("second", symbol)
                .Build("mul");
        return s;
    }

    Symbol Symbol::operator/(const Symbol &symbol) {
        auto s = Symbol("div")
                .SetInput("first", *this)
                .SetInput("second", symbol)
                .Build("div");
        return s;
    }

    const NodePtr &Symbol::GetNode() const {
        return nodePtr;
    }
}

