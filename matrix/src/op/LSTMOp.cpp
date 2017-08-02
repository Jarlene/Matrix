//
// Created by Jarlene on 2017/8/2.
//

#include "matrix/include/op/LSTMOp.h"

namespace matrix {

    template <class T, class Context>
    LSTMOp<T, Context>::LSTMOp(LSTMParam &param) {

    }

    template <class T, class Context>
    bool LSTMOp<T, Context>::Run() {
        return Operator::Run();
    }

    template <class T, class Context>
    void LSTMOp<T, Context>::AsyncRun() {
        Operator::AsyncRun();
    }

    template <class T, class Context>
    bool LSTMOp<T, Context>::RunOnDevice() {
        return false;
    }

    template <class T, class Context>
    LSTMOp<T, Context>::~LSTMOp() {

    }


    template <>
    Operator* CreateOp<cpu>(LSTMParam param, MatrixType type, std::vector<Shape> &in, std::vector<Shape> out) {
        Operator *op = nullptr;
        TYPE_SWITCH(type, DType, {
            op = new LSTMOp<DType, cpu>(param);
        })
        return op;
    }
}