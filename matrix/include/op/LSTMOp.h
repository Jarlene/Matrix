//
// Created by Jarlene on 2017/7/28.
//

#ifndef MATRIX_LSTMOP_H
#define MATRIX_LSTMOP_H

#include "Operator.h"
#include "../base/Tensor.h"

namespace matrix {

    struct LSTMParam : public Parameter {
        LSTMParam(MatrixType matrixType);
    };

    template <class T, class Context>
    class LSTMOp : public Operator {
    SAME_FUNCTION(LSTM);
    DISABLE_COPY_AND_ASSIGN(LSTM);


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


    template <typename Context>
    Operator* CreateOp(LSTMParam param, MatrixType type, std::vector<Shape> &in, std::vector<Shape> out);

}

#endif //MATRIX_LSTMOP_H
