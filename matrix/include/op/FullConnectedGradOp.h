//
// Created by Jarlene on 2017/8/23.
//

#ifndef MATRIX_FULLCONNECTEDGRADOP_H
#define MATRIX_FULLCONNECTEDGRADOP_H


#include "Operator.h"

namespace matrix {




    template <class T, class Context>
    class FullConnectedGradOp : public Operator {
    SAME_FUNCTION(FullConnectedGrad);
    DISABLE_COPY_AND_ASSIGN(FullConnectedGrad);
        INPUT_TAG(PRE_GRAD, OUT, DATA, WEIGHT, BIAS);
    };



    class FullConnectedGradOpProp : public OperatorProperty {
    INIT_OPERATOR_PROPERTY(FullConnectedGradOpProp)

    };
}

REGISTER_OP_PROPERTY(grad_fullConnected, FullConnectedGradOpProp);


#endif //MATRIX_FULLCONNECTEDGRADOP_H
