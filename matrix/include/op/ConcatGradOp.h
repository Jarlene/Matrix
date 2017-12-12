//
// Created by Jarlene on 2017/11/21.
//

#ifndef MATRIX_CONCATGRADOP_H
#define MATRIX_CONCATGRADOP_H

#include "Operator.h"

namespace matrix {

    template <class T, class xpu>
    class ConcatGradOp : public Operator {
    SAME_FUNCTION(ConcatGrad);
    DISABLE_COPY_AND_ASSIGN(ConcatGrad);
        INPUT_TAG(PRE_GRAD, SELF_OUT);
    };


    class ConcatGradOpProp : public OperatorProperty {
    INIT_OPERATOR_PROPERTY(ConcatGradOpProp);

    };
}


REGISTER_OP_PROPERTY(grad_concat, ConcatGradOpProp);

#endif //MATRIX_CONCATGRADOP_H
