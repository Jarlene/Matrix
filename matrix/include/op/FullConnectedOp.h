//
// Created by Jarlene on 2017/7/31.
//

#ifndef MATRIX_FULLCONNECTEDOP_H
#define MATRIX_FULLCONNECTEDOP_H

#include "Operator.h"

namespace matrix {




    template <class T, class Context>
    class FullConnectedOp : public Operator {
    SAME_FUNCTION(FullConnected);
        virtual bool VariableNode(std::function<void(std::initializer_list<Shape *> shapes)> func) override;
    DISABLE_COPY_AND_ASSIGN(FullConnected);
        INPUT_TAG(DATA, WEIGHT, BIAS);
    };

    class FullConnectedOpProp : public OperatorProperty {
    INIT_OPERATOR_PROPERTY(FullConnectedOpProp)

    };
}

REGISTER_OP_PROPERTY(fullConnected, FullConnectedOpProp);

#endif //MATRIX_FULLCONNECTEDOP_H
