//
// Created by Jarlene on 2017/11/24.
//

#ifndef MATRIX_LSTMGRADOP_H
#define MATRIX_LSTMGRADOP_H

#include "Operator.h"
namespace matrix {


    template <class T, class Context>
    class LSTMGradOp : public Operator {
    SAME_FUNCTION(LSTMGrad);
    DISABLE_COPY_AND_ASSIGN(LSTMGrad);
    };

    class LSTMGradOpProp : public OperatorProperty {
    INIT_OPERATOR_PROPERTY(LSTMGradOpProp)
    };
}



REGISTER_OP_PROPERTY(grad_lstm, LSTMGradOpProp);



#endif //MATRIX_LSTMGRADOP_H
