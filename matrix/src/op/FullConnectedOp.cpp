//
// Created by Jarlene on 2017/8/2.
//

#include "matrix/include/op/FullConnectedOp.h"

namespace matrix {

    template <class T, class Context>
    FullConnectedOp<T, Context>::FullConnectedOp(FullConnectedParam &param) {

    }

    template <class T, class Context>
    bool FullConnectedOp<T, Context>::Run() {
        return Operator::Run();
    }

    template <class T, class Context>
    void FullConnectedOp<T, Context>::AsyncRun() {
        Operator::AsyncRun();
    }

    template <class T, class Context>
    FullConnectedOp<T, Context>::~FullConnectedOp() {

    }

    template <class T, class Context>
    bool FullConnectedOp<T, Context>::RunOnDevice() {
        return false;
    }


    template <>
    Operator* CreateOp<cpu>(FullConnectedParam param, MatrixType type, std::vector<Shape> &in, std::vector<Shape> out) {
        Operator *op = nullptr;
        TYPE_SWITCH(type, DType, {
            op = new FullConnectedOp<DType, cpu>(param);
        })
        return op;
    }

}