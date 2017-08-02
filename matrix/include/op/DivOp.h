//
// Created by Jarlene on 2017/7/25.
//

#ifndef MATRIX_DIVOP_H
#define MATRIX_DIVOP_H

#include "Operator.h"

namespace matrix {

    class DivParam : Parameter {

    };

    template <class T, class Context>
    class DivOp : public Operator {
    SAME_FUNCTION(Div);
    DISABLE_COPY_AND_ASSIGN(Div);
    };


    template <typename Context>
    Operator* CreateOp(DivParam param, MatrixType type, std::vector<Shape> &in, std::vector<Shape> out);

}

#endif //MATRIX_DIVOP_H
