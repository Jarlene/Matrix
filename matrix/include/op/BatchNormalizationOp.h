//
// Created by Jarlene on 2017/11/27.
//

#ifndef MATRIX_BATCHNORMALIZATIONOP_H
#define MATRIX_BATCHNORMALIZATIONOP_H



#include "Operator.h"

namespace matrix {



    template <class T, class xpu>
    class BatchNormalizationOp : public Operator {
    SAME_FUNCTION(BatchNormalization);
        virtual bool ShareNodes(std::function<void(std::initializer_list<Shape *> shapes)> func) override;
    DISABLE_COPY_AND_ASSIGN(BatchNormalization);
        INPUT_TAG(DATA, GAMMA, BETA, MEAN, VAR);
    };

    class BatchNormalizationOpProp : public OperatorProperty {
    INIT_OPERATOR_PROPERTY(BatchNormalizationOpProp)
    };

}

REGISTER_OP_PROPERTY(batch_normalization, BatchNormalizationOpProp);


#endif //MATRIX_BATCHNORMALIZATIONOP_H
