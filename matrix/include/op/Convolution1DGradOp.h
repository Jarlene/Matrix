//
// Created by Jarlene on 2017/11/5.
//

#ifndef MATRIX_CONVOLUTION1DGRADOP_H
#define MATRIX_CONVOLUTION1DGRADOP_H

#include "Operator.h"
namespace matrix {





    template<class T, class xpu>
    class Convolution1DGradOp : public Operator {
    SAME_FUNCTION(Convolution1DGrad);
    DISABLE_COPY_AND_ASSIGN(Convolution1DGrad);
        INPUT_TAG(PRE_GRAG, SELF_OUT, DATA, KERNEL, BIAS, COLBUFFER);
    };




    class Convolution1DGradOpProp : public OperatorProperty {
    INIT_OPERATOR_PROPERTY(Convolution1DGradOpProp)

    };

}

REGISTER_OP_PROPERTY(grad_convolution1d, Convolution1DGradOpProp);
#endif //MATRIX_CONVOLUTION1DGRADOP_H
