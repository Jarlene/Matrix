//
// Created by Jarlene on 2017/9/6.
//

#ifndef MATRIX_CONVOLUTION1DOP_H
#define MATRIX_CONVOLUTION1DOP_H

#include "Operator.h"

namespace matrix {



    template <class T, class Context>
    class Convolution1DOp : public Operator {
    SAME_FUNCTION(Convolution1D);
    DISABLE_COPY_AND_ASSIGN(Convolution1D);
        INPUT_TAG(DATA, KERNEL, BIAS);
    };


    class Convolution1DOpProp : public OperatorProperty {
    INIT_OPERATOR_PROPERTY(Convolution1DOpProp)
    };

}

REGISTER_OP_PROPERTY(convolution1d, Convolution1DOpProp);

#endif //MATRIX_CONVOLUTION1DOP_H
