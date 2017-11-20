//
// Created by Jarlene on 2017/11/20.
//

#ifndef MATRIX_CONCATOP_H
#define MATRIX_CONCATOP_H

#include "Operator.h"

namespace matrix {

    template <class T, class Context>
    class ConcatOp : public Operator {
        SAME_FUNCTION(Concat);
        DISABLE_COPY_AND_ASSIGN(Concat);
    };


    class ConcatOpProp : public OperatorProperty {
        INIT_OPERATOR_PROPERTY(ConcatOpProp);

    };
}


REGISTER_OP_PROPERTY(concat, ConcatOpProp);

#endif //MATRIX_CONCATOP_H
