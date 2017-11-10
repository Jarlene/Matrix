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
    };

}


#endif //MATRIX_PARAMETERCOLLECTION_H
