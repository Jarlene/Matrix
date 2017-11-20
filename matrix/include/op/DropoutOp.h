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
        virtual bool ShareNodes(std::function<void(std::initializer_list<Shape *> shapes)> func) override ;
    DISABLE_COPY_AND_ASSIGN(Dropout);
        INPUT_TAG(DATA, MASK);
    };


    class DropoutOpProp : public OperatorProperty {
    INIT_OPERATOR_PROPERTY(DropoutOpProp)

    };
}
REGISTER_OP_PROPERTY(dropout, DropoutOpProp);

#endif //MATRIX_DROPOUTOP_H
