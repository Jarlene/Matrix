//
// Created by Jarlene on 2017/7/31.
//

#ifndef MATRIX_DROPOUTOP_H
#define MATRIX_DROPOUTOP_H

#include "Operator.h"

namespace matrix {

    struct  DropoutParam : public Parameter{
        DropoutParam(MatrixType matrixType);
    };

    template <class T, class Context>
    class DropoutOp : public Operator {
    SAME_FUNCTION(Dropout);
    DISABLE_COPY_AND_ASSIGN(Dropout);
    };

    template <typename Context>
    Operator* CreateOp(DropoutParam param, MatrixType type, std::vector<Shape> &in, std::vector<Shape> out);

}

#endif //MATRIX_DROPOUTOP_H
