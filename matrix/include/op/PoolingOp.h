//
// Created by Jarlene on 2017/7/31.
//

#ifndef MATRIX_POOLOP_H
#define MATRIX_POOLOP_H

#include "Operator.h"

namespace matrix {

    struct PoolingParam : public Parameter {
        PoolingParam(MatrixType matrixType);
    };

    template <class T, class Context>
    class PoolingOp : public Operator {
    SAME_FUNCTION(Pooling);
    DISABLE_COPY_AND_ASSIGN(Pooling);
    };

    template <typename Context>
    Operator* CreateOp(PoolingParam param, MatrixType type, std::vector<Shape> &in, std::vector<Shape> out);

}
#endif //MATRIX_POOLOP_H
