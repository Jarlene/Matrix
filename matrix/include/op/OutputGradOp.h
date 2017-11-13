//
// Created by Jarlene on 2017/11/12.
//

#ifndef MATRIX_OUTPUTGRADOP_H
#define MATRIX_OUTPUTGRADOP_H

#include "Operator.h"

namespace matrix {



    template <class T, class xpu>
    class OutputGradOp : public Operator {
        SAME_FUNCTION(OutputGrad);
        DISABLE_COPY_AND_ASSIGN(OutputGrad);
        INPUT_TAG(PRE_GRAD, OUT, INPUT, LABEL);
    };




    class OutputGradOpProp : public OperatorProperty {
    INIT_OPERATOR_PROPERTY(OutputGradOpProp)

    };

}


REGISTER_OP_PROPERTY(grad_output, OutputGradOpProp);


#endif //MATRIX_OUTPUTGRADOP_H
