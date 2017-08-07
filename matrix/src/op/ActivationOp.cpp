//
// Created by Jarlene on 2017/8/2.
//

#include "matrix/include/op/ActivationOp.h"

namespace matrix {


    template <class T, class Context>
    ActivationOp<T, Context>::ActivationOp(matrix::ActivationParam &param) {

    }

    template <class T, class Context>
    bool ActivationOp<T, Context>::Run() {
        return Operator::Run();
    }

    template <class T, class Context>
    void ActivationOp<T, Context>::AsyncRun() {
        Operator::AsyncRun();
    }

    template <class T, class Context>
    bool ActivationOp<T, Context>::RunOnDevice() {
        return false;
    }

    template <class T, class Context>
    ActivationOp<T, Context>::~ActivationOp() {

    }


    template <>
    Operator* CreateOp<cpu>(ActivationParam param, MatrixType type, std::vector<Shape> &in, std::vector<Shape> out) {
        Operator *op = nullptr;
        TYPE_SWITCH(type, DType, {
            op = new ActivationOp<DType, cpu>(param);
        })
        return op;
    }

    ActivationParam::ActivationParam(MatrixType matrixType) : Parameter(matrixType) {

    }
}