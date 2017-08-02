//
// Created by Jarlene on 2017/7/25.
//

#ifndef MATRIX_SUBOP_H
#define MATRIX_SUBOP_H

#include "Operator.h"

namespace matrix {

    class SubParam : Parameter {

    };

    template <class T, class Context>
    class SubOp : public Operator {
    SAME_FUNCTION(Sub);
    DISABLE_COPY_AND_ASSIGN(Sub);
    };

    template <typename Context>
    Operator* CreateOp(SubParam param, MatrixType type, std::vector<Shape> &in, std::vector<Shape> out);

}

#endif //MATRIX_SUBOP_H
