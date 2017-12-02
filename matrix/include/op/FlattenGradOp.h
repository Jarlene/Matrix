//
// Created by Jarlene on 2017/11/9.
//

#ifndef MATRIX_FLATTENGRADOP_H
#define MATRIX_FLATTENGRADOP_H


#include "Operator.h"
namespace matrix {




    template <class T, class xpu>
    class FlattenGradOp : public Operator {
    SAME_FUNCTION(FlattenGrad);
    DISABLE_COPY_AND_ASSIGN(FlattenGrad);
    };



    class FlattenGradOpProp : public OperatorProperty {
    INIT_OPERATOR_PROPERTY(FlattenGradOpProp)
    };

}

REGISTER_OP_PROPERTY(grad_flatten, FlattenGradOpProp);


#endif //MATRIX_FLATTENGRADOP_H
