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
    Operator* CreateOp(LSTMParam &param);


    class LSTMOpProp : public OperatorProperty {
    public:
        LSTMOpProp();
        LSTMOpProp(const MatrixType &type);
        ~LSTMOpProp();
        virtual void InferShape(std::vector<Shape> &inShape, std::vector<Shape> &outShape);
        virtual Operator* CreateOperator(Context context, std::vector<Blob> &input, std::vector<Blob> &output, std::vector<Shape> &inShape, std::vector<Shape> &outShape) ;
    private:
        LSTMParam* param;
    };
}

REGISTER_OP_PROPERTY(lstm, LSTMOpProp);

#endif //MATRIX_LSTMOP_H
