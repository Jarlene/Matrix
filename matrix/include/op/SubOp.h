//
// Created by Jarlene on 2017/7/25.
//

#ifndef MATRIX_SUBOP_H
#define MATRIX_SUBOP_H

#include "Operator.h"

namespace matrix {



    template <class T, class xpu>
    class SubOp : public Operator {
    SAME_FUNCTION(Sub);
    DISABLE_COPY_AND_ASSIGN(Sub);
        INPUT_TAG(INPUT1,INPUT2);
    };



    class SubOpProp : public OperatorProperty {
    INIT_OPERATOR_PROPERTY(SubOpProp)

    };
}

REGISTER_OP_PROPERTY(sub, SubOpProp);

#endif //MATRIX_SUBOP_H
