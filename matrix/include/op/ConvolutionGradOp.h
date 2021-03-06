//
// Created by Jarlene on 2017/9/8.
//

#ifndef MATRIX_CONVOLUTIONGRADOP_H
#define MATRIX_CONVOLUTIONGRADOP_H


#include "Operator.h"

namespace matrix {



    template <class T, class xpu>
    class ConvolutionGradOp : public Operator {
    SAME_FUNCTION(ConvolutionGrad);
        virtual bool ShareNodes(std::function<void(std::initializer_list<Shape *> shapes)> func) override;
    DISABLE_COPY_AND_ASSIGN(ConvolutionGrad);
        INPUT_TAG(PRE_GRAG, SELF_OUT, DATA, KERNEL, BIAS, COLBUFFER);
    };


    class ConvolutionOpGradProp : public OperatorProperty {
    INIT_OPERATOR_PROPERTY(ConvolutionOpGradProp)

    };

}

REGISTER_OP_PROPERTY(grad_convolution, ConvolutionOpGradProp);

#endif //MATRIX_CONVOLUTIONGRADOP_H
