//
// Created by  Jarlene on 2017/8/9.
//

#ifndef MATRIX_APPLYGRADOP_H
#define MATRIX_APPLYGRADOP_H


#include "Operator.h"

namespace matrix {



    template <class T, class xpu>
    class UpdateOp : public Operator {
    SAME_FUNCTION(Update);
    DISABLE_COPY_AND_ASSIGN(Update);
        INPUT_TAG(VARIABLE, GRAD_VARIABLE);
    };




    class UpdateOpProp : public OperatorProperty {
    INIT_OPERATOR_PROPERTY(UpdateOpProp)

    };
}

REGISTER_OP_PROPERTY(applyGrad, UpdateOpProp);

#endif //MATRIX_APPLYGRADOP_H
