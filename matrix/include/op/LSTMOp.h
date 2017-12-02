//
// Created by Jarlene on 2017/7/28.
//

#ifndef MATRIX_LSTMOP_H
#define MATRIX_LSTMOP_H

#include "Operator.h"
#include "../base/Tensor.h"

namespace matrix {



    template <class T, class xpu>
    class LSTMOp : public Operator {
    SAME_FUNCTION(LSTM);
        virtual bool VariableNode(std::function<void(std::initializer_list<Shape *> shapes)> func) override ;
        virtual bool ShareNodes(std::function<void(std::initializer_list<Shape *> shapes)> func) override ;
    DISABLE_COPY_AND_ASSIGN(LSTM);
        INPUT_TAG(INPUT, CELL, WEIGHT, BIAS);

    };

    template <class T>
    class LSTMState : public State {
    public:
        Tensor<T> h, c, u, i ,f ,o;
        Tensor<T> cTanh;
        Tensor<T> maskXt, maskAt, maskHt; // dropout
        Tensor<T> delth, deltc, deltx, delta; // bp
        virtual void clear();
    };




    class LSTMOpProp : public OperatorProperty {
    INIT_OPERATOR_PROPERTY(LSTMOpProp)
    };
}

REGISTER_OP_PROPERTY(lstm, LSTMOpProp);

#endif //MATRIX_LSTMOP_H
