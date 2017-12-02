//
// Created by Jarlene on 2017/9/6.
//

#ifndef MATRIX_NAVIECONVOLUTIONOP_H
#define MATRIX_NAVIECONVOLUTIONOP_H

#include "Operator.h"

namespace matrix {



    template <class T, class xpu>
    class NaiveConvolutionOp : public Operator {
        SAME_FUNCTION(NaiveConvolution);
        DISABLE_COPY_AND_ASSIGN(NaiveConvolution);
        INPUT_TAG(DATA, KERNEL, BIAS);
    };


    class NaiveConvolutionOpProp : public OperatorProperty {
    INIT_OPERATOR_PROPERTY(NaiveConvolutionOpProp)

    };

}

REGISTER_OP_PROPERTY(navie_convolution, NaiveConvolutionOpProp);

#endif //MATRIX_NAVIECONVOLUTIONOP_H
