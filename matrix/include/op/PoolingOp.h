//
// Created by Jarlene on 2017/7/31.
//

#ifndef MATRIX_POOLOP_H
#define MATRIX_POOLOP_H

#include "Operator.h"

namespace matrix {


    template <class T, class Context>
    class PoolingOp : public Operator {
    SAME_FUNCTION(Pooling);
    DISABLE_COPY_AND_ASSIGN(Pooling);
    };



    class PoolingOpProp : public OperatorProperty {
    INIT_OPERATOR_PROPERTY(PoolingOpProp)

    };
}

REGISTER_OP_PROPERTY(pooling, PoolingOpProp);

#endif //MATRIX_POOLOP_H
