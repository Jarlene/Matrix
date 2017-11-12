//
// Created by Jarlene on 2017/7/25.
//

#ifndef MATRIX_DIVOP_H
#define MATRIX_DIVOP_H

#include "Operator.h"

namespace matrix {



    template <class T, class Context>
    class DivOp : public Operator {
    SAME_FUNCTION(Div);
    DISABLE_COPY_AND_ASSIGN(Div);
    };




    class DivOpProp : public OperatorProperty {
    INIT_OPERATOR_PROPERTY(DivOpProp)

    };

}
REGISTER_OP_PROPERTY(div, DivOpProp);

#endif //MATRIX_DIVOP_H
