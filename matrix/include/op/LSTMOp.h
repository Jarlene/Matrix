//
// Created by Jarlene on 2017/7/28.
//

#ifndef MATRIX_LSTMOP_H
#define MATRIX_LSTMOP_H

#include "Operator.h"
#include "../base/Tensor.h"

namespace matrix {


    template <class T, class Context>
    class LSTMOp : public Operator {
    public:


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

}

#endif //MATRIX_LSTMOP_H
