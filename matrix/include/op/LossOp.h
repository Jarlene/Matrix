//
// Created by Jarlene on 2017/7/31.
//

#ifndef MATRIX_LOSSOP_H
#define MATRIX_LOSSOP_H


#include "Operator.h"

namespace matrix {



    template <class T, class Context>
    class LossOp : public Operator {
    SAME_FUNCTION(Loss);
    DISABLE_COPY_AND_ASSIGN(Loss);
        INPUT_TAG(DATA, LABEL);
    };



    class LossOpProp : public OperatorProperty {
    INIT_OPERATOR_PROPERTY(LossOpProp)

    };
}

REGISTER_OP_PROPERTY(loss, LossOpProp);

#endif //MATRIX_LOSSOP_H
