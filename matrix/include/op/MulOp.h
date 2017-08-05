//
// Created by Jarlene on 2017/7/25.
//

#ifndef MATRIX_MULOP_H
#define MATRIX_MULOP_H

#include "Operator.h"

namespace matrix {

    struct MulParam : public Parameter {
        MulParam(MatrixType matrixType);
    };

    template <class T, class Context>
    class MulOp : public Operator {
    SAME_FUNCTION(Mul);
    DISABLE_COPY_AND_ASSIGN(Mul);
    };

    template <typename Context>
    Operator* CreateOp(MulParam param, MatrixType type, std::vector<Shape> &in, std::vector<Shape> out);

}
#endif //MATRIX_MULOP_H
