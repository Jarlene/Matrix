//
// Created by Jarlene on 2017/7/25.
//

#ifndef MATRIX_ADDOP_H
#define MATRIX_ADDOP_H

#include "Operator.h"

namespace matrix {



    template <class T, class xpu>
    class AddOp : public Operator {
    SAME_FUNCTION(Add);
    DISABLE_COPY_AND_ASSIGN(Add);
        INPUT_TAG(INPUT1, INPUT2);
    };




    class AddOpProp : public OperatorProperty {
        INIT_OPERATOR_PROPERTY(AddOpProp)
    };

}

REGISTER_OP_PROPERTY(add, AddOpProp);


#endif //MATRIX_ADDOP_H
