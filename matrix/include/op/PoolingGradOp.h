//
// Created by Jarlene on 2017/11/6.
//

#ifndef MATRIX_POOLINGGRADOP_H
#define MATRIX_POOLINGGRADOP_H

#include "Operator.h"

namespace matrix {




    template <class T, class xpu>
    class PoolingGradOp : public Operator {
    SAME_FUNCTION(PoolingGrad);
    DISABLE_COPY_AND_ASSIGN(PoolingGrad)
        INPUT_TAG(PRE_GRAG, SELF_OUT, INPUT, MAX_INDEX);
    };



    class PoolingGradOpProp : public OperatorProperty {
    INIT_OPERATOR_PROPERTY(PoolingGradOpProp)
    };
}



REGISTER_OP_PROPERTY(grad_pooling, PoolingGradOpProp);



#endif //MATRIX_POOLINGGRADOP_H
