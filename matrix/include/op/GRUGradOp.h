//
// Created by Jarlene on 2017/7/28.
//

#ifndef MATRIX_GRUGRADOP_H
#define MATRIX_GRUGRADOP_H

#include "Operator.h"

namespace matrix {



    template <class T, class xpu>
    class GRUGradOp : public Operator {
    SAME_FUNCTION(GRUGrad);
    DISABLE_COPY_AND_ASSIGN(GRUGrad);
    };



    class GRUGradOpProp : public OperatorProperty {
    INIT_OPERATOR_PROPERTY(GRUGradOpProp)

    };
}
REGISTER_OP_PROPERTY(gru_grad, GRUGradOpProp);
#endif //MATRIX_GRUGRADOP_H
