//
// Created by Jarlene on 2017/7/18.
//

#ifndef MATRIX_SYMBOL_H
#define MATRIX_SYMBOL_H

#include <string>
#include <unordered_map>
#include "matrix/include/utils/Any.h"
#include "matrix/include/base/Node.h"


namespace matrix {
    class Symbol {
    public:

        Symbol() = default;

        Symbol(const Symbol &symbol);

        explicit Symbol(const std::string &name);

        Symbol &SetInput(const std::string &name, const Symbol &symbol);

        Symbol &SetParam(const std::string &name, const Any &value);

        template<class T>
        Symbol &SetParam(const std::string &name, const T &value) {
            return SetParam(name, Any(value));
        }

        Symbol &On(const RunMode &mode = kCpu);

        Symbol &Build();

        Symbol &operator=(const Symbol &symbol);

        Symbol operator+(const Symbol &symbol);

        Symbol operator-(const Symbol &symbol);

        Symbol operator*(const Symbol &symbol);

        Symbol operator/(const Symbol &symbol);

    protected:
        friend class Graph;
        const NodePtr &GetNode() const ;

    protected:
       NodePtr nodePtr;
    };
}



#endif //MATRIX_SYMBOL_H
