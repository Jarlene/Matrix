//
// Created by Jarlene on 2017/7/28.
//

#ifndef MATRIX_LSTMOP_H
#define MATRIX_LSTMOP_H

#include "Operator.h"
#include "../base/Tensor.h"

namespace matrix {

    class LSTMParam : Parameter {

    };

    template <class T, class Context>
    class LSTMOp : public Operator {
    public:
        explicit LSTMOp(LSTMParam &param);

        virtual bool Run() override ;

        virtual void AsyncRun() override ;

        virtual ~LSTMOp();

        virtual bool RunOnDevice() override ;

    DISABLE_COPY_AND_ASSIGN(LSTMOp);


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
