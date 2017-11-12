//
// Created by Jarlene on 2017/11/9.
//

#ifndef MATRIX_RNNOP_H
#define MATRIX_RNNOP_H


#include "Operator.h"

namespace matrix {


    
    template <class T, class Context>
    class RNNOp : public Operator {
    SAME_FUNCTION(RNN);
    DISABLE_COPY_AND_ASSIGN(RNN);
    };


    template <class T>
    class RNNState : public State {
    public:

    };



    class RNNOpProp : public OperatorProperty {
    INIT_OPERATOR_PROPERTY(RNNOpProp)
    };
}


REGISTER_OP_PROPERTY(rnn, RNNOpProp);




#endif //MATRIX_RNNOP_H
