//
// Created by Jarlene on 2017/12/7.
//

#ifndef MATRIX_SUBGRADOP_H
#define MATRIX_SUBGRADOP_H




#include "Operator.h"

namespace matrix {



    template <class T, class xpu>
    class SubGradOp : public Operator {
    SAME_FUNCTION(SubGrad);
    DISABLE_COPY_AND_ASSIGN(SubGrad);
        INPUT_TAG(PRE_GRAD, SELF_OUT, INPUT1,INPUT2);
    };



    class SubGradOpProp : public OperatorProperty {
    INIT_OPERATOR_PROPERTY(SubGradOpProp)

    };
}

REGISTER_OP_PROPERTY(grad_sub, SubGradOpProp);




#endif //MATRIX_SUBGRADOP_H
