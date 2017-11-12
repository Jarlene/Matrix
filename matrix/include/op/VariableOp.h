//
// Created by Jarlene on 2017/8/9.
//

#ifndef MATRIX_VARIABLEOP_H
#define MATRIX_VARIABLEOP_H

#include "Operator.h"

namespace matrix {



    template <class T, class xpu>
    class VariableOp : public Operator {
    SAME_FUNCTION(Variable);
    DISABLE_COPY_AND_ASSIGN(Variable);
        INPUT_TAG(INPUT);
    };




    class VariableOpProp : public OperatorProperty {
    INIT_OPERATOR_PROPERTY(VariableOpProp)

    };
}
REGISTER_OP_PROPERTY(variable, VariableOpProp);

#endif //MATRIX_VARIABLEOP_H
