//
// Created by Jarlene on 2017/11/21.
//

#ifndef MATRIX_TRANSPOSEOP_H
#define MATRIX_TRANSPOSEOP_H



#include "Operator.h"

namespace matrix {



    template <class T, class xpu>
    class TransposeOp : public Operator {
    SAME_FUNCTION(Transpose);
    DISABLE_COPY_AND_ASSIGN(Transpose);
    };



    class TransposeOpProp : public OperatorProperty {
    INIT_OPERATOR_PROPERTY(TransposeOpProp)

    };
}

REGISTER_OP_PROPERTY(transpose, TransposeOpProp);

#endif //MATRIX_TRANSPOSEOP_H
