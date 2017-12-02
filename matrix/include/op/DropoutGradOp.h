//
// Created by 郑珊 on 2017/12/2.
//

#ifndef MATRIX_DROPOUTGRADOP_H
#define MATRIX_DROPOUTGRADOP_H




#include "Operator.h"

namespace matrix {


    template <class T, class xpu>
    class DropoutGradOp : public Operator {
    SAME_FUNCTION(DropoutGrad);
    DISABLE_COPY_AND_ASSIGN(DropoutGrad);
        INPUT_TAG(PRE_GRAD, SELF_OUT, DATA, MASK);
    };


    class DropoutGradOpProp : public OperatorProperty {
    INIT_OPERATOR_PROPERTY(DropoutGradOpProp)

    };
}
REGISTER_OP_PROPERTY(grad_dropout, DropoutGradOpProp);

#endif //MATRIX_DROPOUTGRADOP_H
