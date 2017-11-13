//
// Created by Jarlene on 2017/8/9.
//

#ifndef MATRIX_OUTPUTOP_H
#define MATRIX_OUTPUTOP_H

#include "Operator.h"

namespace matrix {



    template <class T, class xpu>
    class OutputOp : public Operator {
    SAME_FUNCTION(Output);
    DISABLE_COPY_AND_ASSIGN(Output);
        INPUT_TAG(DATA, LABEL);
    };




    class OutputOpProp : public OperatorProperty {
    INIT_OPERATOR_PROPERTY(OutputOpProp)


    };
}

REGISTER_OP_PROPERTY(output, OutputOpProp);

#endif //MATRIX_OUTPUTOP_H
