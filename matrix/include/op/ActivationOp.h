//
// Created by Jarlene on 2017/7/31.
//

#ifndef MATRIX_ACTIVATIONOP_H
#define MATRIX_ACTIVATIONOP_H

#include "Operator.h"

namespace matrix {



    template <class T, class xpu>
    class ActivationOp : public Operator {
    SAME_FUNCTION(Activation);
    DISABLE_COPY_AND_ASSIGN(Activation);
        INPUT_TAG(DATA);
    };


    class ActivationOpProp : public OperatorProperty {
    INIT_OPERATOR_PROPERTY(ActivationOpProp)

    };

}

REGISTER_OP_PROPERTY(activation, ActivationOpProp);

#endif //MATRIX_ACTIVATIONOP_H
