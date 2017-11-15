//
// Created by Jarlene on 2017/11/8.
//

#ifndef MATRIX_PARAMETERCOLLECTION_H
#define MATRIX_PARAMETERCOLLECTION_H

#include "Symbol.h"

namespace matrix {

    class ParameterCollection {
    public:
        void add(const Symbol &express);
        void add_param(const Symbol &symbol);
        void add_input(const Symbol &symbol);

    private:
        std::vector<NodePtr> express;
        std::vector<NodePtr> params;
        std::vector<NodePtr> inputs;
    };

}


#endif //MATRIX_PARAMETERCOLLECTION_H
