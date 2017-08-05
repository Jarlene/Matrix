//
// Created by Jarlene on 2017/8/2.
//

#include "matrix/include/op/DropoutOp.h"

namespace matrix {

    template <class T, class Context>
    DropoutOp<T, Context>::DropoutOp(DropoutParam &param) {

    }

    template <class T, class Context>
    bool DropoutOp<T, Context>::Run() {
        return Operator::Run();
    }

    template <class T, class Context>
    void DropoutOp<T, Context>::AsyncRun() {
        Operator::AsyncRun();
    }

    template <class T, class Context>
    bool DropoutOp<T, Context>::RunOnDevice() {
        return false;
    }

    template <class T, class Context>
    DropoutOp<T, Context>::~DropoutOp() {

    }


    template <class T, class Context>
    bool DropoutOp<T, Context>::InferShape() {
        return false;
    }

    template <>
    Operator* CreateOp<cpu>(DropoutParam param, MatrixType type, std::vector<Shape> &in, std::vector<Shape> out) {
        Operator *op = nullptr;
        TYPE_SWITCH(type, DType, {
            op = new DropoutOp<DType, cpu>(param);
        })
        return op;
    }

    DropoutParam::DropoutParam(MatrixType matrixType) : Parameter(matrixType) {

    }
}