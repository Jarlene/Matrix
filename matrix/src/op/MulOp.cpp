//
// Created by Jarlene on 2017/8/2.
//

#include "matrix/include/op/MulOp.h"

namespace matrix {


    template <class T, class Context>
    MulOp<T, Context>::MulOp(MulParam &param) {

    }

    template <class T, class Context>
    bool MulOp<T, Context>::Run() {
        return Operator::Run();
    }

    template <class T, class Context>
    void MulOp<T, Context>::AsyncRun() {
        Operator::AsyncRun();
    }

    template <class T, class Context>
    bool MulOp<T, Context>::RunOnDevice() {
        return false;
    }

    template <class T, class Context>
    MulOp<T, Context>::~MulOp() {

    }



    template <>
    Operator* CreateOp<CPU>(MulParam param, MatrixType type, std::vector<Shape> &in, std::vector<Shape> out) {
        Operator *op = nullptr;
        TYPE_SWITCH(type, DType, {
            op = new MulOp<DType, CPU>(param);
        })
        return op;
    }

    MulParam::MulParam(MatrixType matrixType) : Parameter(matrixType) {

    }
}