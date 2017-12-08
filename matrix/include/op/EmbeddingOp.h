//
// Created by Jarlene on 2017/12/8.
//

#ifndef MATRIX_EMBEDDINGOP_H
#define MATRIX_EMBEDDINGOP_H




#include "Operator.h"

namespace matrix {




    template <class T, class xpu>
    class EmbeddingOp : public Operator {
    SAME_FUNCTION(Embedding);
    DISABLE_COPY_AND_ASSIGN(Embedding);
    };



    class EmbeddingOpProp : public OperatorProperty {
    INIT_OPERATOR_PROPERTY(EmbeddingOpProp)

    };
}

REGISTER_OP_PROPERTY(embedding, EmbeddingOpProp);



#endif //MATRIX_EMBEDDINGOP_H
