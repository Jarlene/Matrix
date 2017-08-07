//
// Created by Jarlene on 2017/8/2.
//

#include "matrix/include/op/LossOp.h"


namespace matrix {

    template <class T, class Context>
    LossOp<T, Context>::LossOp(LossParam &param) {

    }

    template <class T, class Context>
    bool LossOp<T, Context>::Run() {
        return Operator::Run();
    }

    template <class T, class Context>
    void LossOp<T, Context>::AsyncRun() {
        Operator::AsyncRun();
    }

    template <class T, class Context>
    LossOp<T, Context>::~LossOp() {

    }

    template <class T, class Context>
    bool LossOp<T, Context>::RunOnDevice() {
        return false;
    }




    template <>
    Operator* CreateOp<cpu>(LossParam param, MatrixType type, std::vector<Shape> &in, std::vector<Shape> out) {
        Operator *op = nullptr;
        TYPE_SWITCH(type, DType, {
            op = new LossOp<DType, cpu>(param);
        })
        return op;
    }

    LossParam::LossParam(MatrixType matrixType) : Parameter(matrixType) {

    }
}