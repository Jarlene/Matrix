//
// Created by Jarlene on 2017/11/5.
//

#ifndef MATRIX_LOSSGRADOP_H
#define MATRIX_LOSSGRADOP_H

#include "Operator.h"


namespace matrix {



    template <class T, class Context>
    class LossGradOp : public Operator{
    SAME_FUNCTION(LossGrad);
    DISABLE_COPY_AND_ASSIGN(LossGrad);
        INPUT_TAG(DATA, LABEL);
    };



    class LossGradOpProp : public OperatorProperty {
    INIT_OPERATOR_PROPERTY(LossGradOpProp)

    };
}


REGISTER_OP_PROPERTY(grad_loss, LossGradOpProp);


#endif //MATRIX_LOSSGRADOP_H
