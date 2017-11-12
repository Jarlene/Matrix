//
// Created by Jarlene on 2017/7/31.
//

#ifndef MATRIX_DROPOUTOP_H
#define MATRIX_DROPOUTOP_H

#include "Operator.h"

namespace matrix {


    template <class T, class Context>
    class DropoutOp : public Operator {
    SAME_FUNCTION(Dropout);
    DISABLE_COPY_AND_ASSIGN(Dropout);
    };


    class DropoutOpProp : public OperatorProperty {
    INIT_OPERATOR_PROPERTY(DropoutOpProp)

    };
}
REGISTER_OP_PROPERTY(dropout, DropoutOpProp);

#endif //MATRIX_DROPOUTOP_H
