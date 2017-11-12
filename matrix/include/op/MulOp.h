//
// Created by Jarlene on 2017/7/25.
//

#ifndef MATRIX_MULOP_H
#define MATRIX_MULOP_H

#include "Operator.h"

namespace matrix {



    template <class T, class Context>
    class MulOp : public Operator {
    SAME_FUNCTION(Mul);
    DISABLE_COPY_AND_ASSIGN(Mul);
        INPUT_TAG(INPUT1, INPUT2);
    };


    class MulOpProp : public OperatorProperty {
    INIT_OPERATOR_PROPERTY(MulOpProp)

    };
}

REGISTER_OP_PROPERTY(mul, MulOpProp);
#endif //MATRIX_MULOP_H
