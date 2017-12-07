//
// Created by Jarlene on 2017/12/7.
//

#ifndef MATRIX_DIVGRADOP_H
#define MATRIX_DIVGRADOP_H


#include "Operator.h"

namespace matrix {



    template <class T, class Context>
    class DivGradOp : public Operator {
    SAME_FUNCTION(DivGrad);
    DISABLE_COPY_AND_ASSIGN(DivGrad);
        INPUT_TAG(PRE_GRAD, SELF_OUT, INPUT1, INPUT2);
    };




    class DivGradOpProp : public OperatorProperty {
    INIT_OPERATOR_PROPERTY(DivGradOpProp)
    };

}
REGISTER_OP_PROPERTY(grad_div, DivGradOpProp);

#endif //MATRIX_DIVGRADOP_H
