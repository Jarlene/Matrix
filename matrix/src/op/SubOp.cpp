//
// Created by Jarlene on 2017/8/2.
//

#include "matrix/include/op/SubOp.h"

namespace matrix {
    template <class T, class Context>
    SubOp<T, Context>::SubOp(SubParam &param) {

    }

    template <class T, class Context>
    bool SubOp<T, Context>::Run() {
        return Operator::Run();
    }

    template <class T, class Context>
    void SubOp<T, Context>::AsyncRun() {
        Operator::AsyncRun();
    }

    template <class T, class Context>
    bool SubOp<T, Context>::RunOnDevice() {
        return false;
    }

    template <class T, class Context>
    SubOp<T, Context>::~SubOp() {

    }


    template <>
    Operator* CreateOp<cpu>(SubParam param, MatrixType type, std::vector<Shape> &in, std::vector<Shape> out) {
        Operator *op = nullptr;
        TYPE_SWITCH(type, DType, {
            op = new SubOp<DType, cpu>(param);
        })
        return op;
    }
}