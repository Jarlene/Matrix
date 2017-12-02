//
// Created by Jarlene on 2017/7/28.
//

#ifndef MATRIX_GRUOP_H
#define MATRIX_GRUOP_H

#include "Operator.h"

namespace matrix {



    template <class T, class xpu>
    class GRUOp : public Operator {
    SAME_FUNCTION(GRU);
    DISABLE_COPY_AND_ASSIGN(GRU);
    };



    class GRUOpProp : public OperatorProperty {
    INIT_OPERATOR_PROPERTY(GRUOpProp)

    };
}
REGISTER_OP_PROPERTY(gru, GRUOpProp);
#endif //MATRIX_GRUOP_H
