//
// Created by Jarlene on 2017/7/28.
//

#ifndef MATRIX_GRUOP_H
#define MATRIX_GRUOP_H

#include "Operator.h"

namespace matrix {

    class GRUParam : Parameter {

    };

    template <class T, class Context>
    class GRUOp : public Operator {
    SAME_FUNCTION(GRU);
    DISABLE_COPY_AND_ASSIGN(GRU);
    };

    template <typename Context>
    Operator* CreateOp(GRUParam param, MatrixType type, std::vector<Shape> &in, std::vector<Shape> out);

}

#endif //MATRIX_GRUOP_H
