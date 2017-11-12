//
// Created by Jarlene on 2017/8/23.
//

#ifndef MATRIX_ACTIVATIONGRADOP_H
#define MATRIX_ACTIVATIONGRADOP_H

#include "Operator.h"

namespace matrix {




    template <class T, class Context>
    class ActivationGradOp : public Operator {
        SAME_FUNCTION(ActivationGrad);
        DISABLE_COPY_AND_ASSIGN(ActivationGrad);
        INPUT_TAG(PRE_GRAD, OUT, INPUT);
    };



    class ActivationOpGradProp : public OperatorProperty {
    INIT_OPERATOR_PROPERTY(ActivationOpGradProp)

    };


}

REGISTER_OP_PROPERTY(grad_activation, ActivationOpGradProp);


#endif //MATRIX_ACTIVATIONGRADOP_H
