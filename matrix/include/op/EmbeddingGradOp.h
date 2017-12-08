//
// Created by Jarlene on 2017/12/8.
//

#ifndef MATRIX_EMBEDDINGGRADOP_H
#define MATRIX_EMBEDDINGGRADOP_H


class EmbeddingGradOp {

};


#include "Operator.h"

namespace matrix {




    template <class T, class xpu>
    class EmbeddingGradOp : public Operator {
    SAME_FUNCTION(EmbeddingGrad);
    DISABLE_COPY_AND_ASSIGN(EmbeddingGrad);
    };



    class EmbeddingGradOpProp : public OperatorProperty {
    INIT_OPERATOR_PROPERTY(EmbeddingGradOpProp)

    };
}

REGISTER_OP_PROPERTY(grad_embedding, EmbeddingGradOpProp);




#endif //MATRIX_EMBEDDINGGRADOP_H
