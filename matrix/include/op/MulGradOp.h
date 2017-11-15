//
// Created by Jarlene on 2017/11/15.
//

#ifndef MATRIX_MULGRADOP_H
#define MATRIX_MULGRADOP_H

#include "Operator.h"

namespace matrix {



    template <class T, class Context>
    class MulGradOp : public Operator {
    SAME_FUNCTION(MulGrad);
    DISABLE_COPY_AND_ASSIGN(MulGrad);
        INPUT_TAG(PRE_GRAD, SELF_OUT, INPUT1, INPUT2);
    };


    class MulGradOpProp : public OperatorProperty {
    INIT_OPERATOR_PROPERTY(MulGradOpProp)

    };
}

REGISTER_OP_PROPERTY(grad_mul, MulGradOpProp);

#endif //MATRIX_MULGRADOP_H
