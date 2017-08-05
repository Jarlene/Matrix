//
// Created by Jarlene on 2017/8/2.
//

#include "matrix/include/op/PoolingOp.h"

namespace matrix {

    template <class T, class Context>
    PoolingOp<T, Context>::PoolingOp(PoolingParam &param) {

    }

    template <class T, class Context>
    bool PoolingOp<T, Context>::Run() {
        return Operator::Run();
    }

    template <class T, class Context>
    void PoolingOp<T, Context>::AsyncRun() {
        Operator::AsyncRun();
    }

    template <class T, class Context>
    bool PoolingOp<T, Context>::RunOnDevice() {
        return false;
    }

    template <class T, class Context>
    PoolingOp<T, Context>::~PoolingOp() {

    }


    template <class T, class Context>
    bool PoolingOp<T, Context>::InferShape() {
        return false;
    }

    template <>
    Operator* CreateOp<cpu>(PoolingParam param, MatrixType type, std::vector<Shape> &in, std::vector<Shape> out) {
        Operator *op = nullptr;
        TYPE_SWITCH(type, DType, {
            op = new PoolingOp<DType, cpu>(param);
        })
        return op;
    }

    PoolingParam::PoolingParam(MatrixType matrixType) : Parameter(matrixType) {

    }
}