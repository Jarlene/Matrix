//
// Created by Jarlene on 2017/7/25.
//

#ifndef MATRIX_ADDOP_H
#define MATRIX_ADDOP_H

#include "Operator.h"

namespace matrix {

    class AddParam : Parameter {

    };

    template <class T, class Context>
    class AddOp : public Operator {
    SAME_FUNCTION(Add);
    DISABLE_COPY_AND_ASSIGN(Add);
    };


    template <typename Context>
    Operator* CreateOp(AddParam param, MatrixType type, std::vector<Shape> &in, std::vector<Shape> out);



}

#endif //MATRIX_ADDOP_H
