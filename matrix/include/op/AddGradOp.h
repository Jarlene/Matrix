//
// Created by Jarlene on 2017/8/23.
//

#ifndef MATRIX_ADDGRADOP_H
#define MATRIX_ADDGRADOP_H


#include "Operator.h"

namespace matrix {



    template <class T, class xpu>
    class AddGradOp : public Operator {
    SAME_FUNCTION(AddGrad);
    DISABLE_COPY_AND_ASSIGN(AddGrad);
        INPUT_TAG(PRE_GRAD, OUT, INPUT1, INPUT2);
    };



    class AddGradOpProp : public OperatorProperty {
    INIT_OPERATOR_PROPERTY(AddGradOpProp)
    };

}

REGISTER_OP_PROPERTY(grad_add, AddGradOpProp);

#endif //MATRIX_ADDGRADOP_H
