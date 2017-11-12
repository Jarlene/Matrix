//
// Created by Jarlene on 2017/7/31.
//

#ifndef MATRIX_CONVOLUTIONOP_H
#define MATRIX_CONVOLUTIONOP_H

#include "Operator.h"

namespace matrix {



    template <class T, class Context>
    class ConvolutionOp : public Operator {
    SAME_FUNCTION(Convolution);
        virtual void VariableNode(std::function<void(std::initializer_list<Shape *> shapes)> func) override;
    DISABLE_COPY_AND_ASSIGN(Convolution);
        INPUT_TAG(DATA, KERNEL, BIAS, COLBUFFER);
    };



    class ConvolutionOpProp : public OperatorProperty {
    INIT_OPERATOR_PROPERTY(ConvolutionOpProp)

    };

}

REGISTER_OP_PROPERTY(convolution, ConvolutionOpProp);

#endif //MATRIX_CONVOLUTIONOP_H
