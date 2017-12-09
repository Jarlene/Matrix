//
// Created by Jarlene on 2017/11/5.
//

#ifndef MATRIX_FLATTENOP_H
#define MATRIX_FLATTENOP_H

#include "Operator.h"

namespace matrix {




    template <class T, class xpu>
    class FlattenOp : public Operator {
    SAME_FUNCTION(Flatten);
    DISABLE_COPY_AND_ASSIGN(Flatten);
        INPUT_TAG(INPUT);
    };



    class FlattenOpProp : public OperatorProperty {
    INIT_OPERATOR_PROPERTY(FlattenOpProp)

    };
}

REGISTER_OP_PROPERTY(flatten, FlattenOpProp);



#endif //MATRIX_FLATTENOP_H
