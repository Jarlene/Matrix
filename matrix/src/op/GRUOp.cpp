//
// Created by Jarlene on 2017/8/2.
//

#include "matrix/include/op/GRUOp.h"

namespace matrix {

    template <class T, class Context>
    GRUOp<T, Context>::GRUOp(GRUParam &param) {

    }

    template <class T, class Context>
    bool GRUOp<T, Context>::Run() {
        return Operator::Run();
    }

    template <class T, class Context>
    void GRUOp<T, Context>::AsyncRun() {
        Operator::AsyncRun();
    }

    template <class T, class Context>
    bool GRUOp<T, Context>::RunOnDevice() {
        return false;
    }

    template <class T, class Context>
    GRUOp<T, Context>::~GRUOp() {

    }

    template <class T, class Context>
    bool GRUOp<T, Context>::InferShape() {
        return false;
    }

    template <>
    Operator* CreateOp<cpu>(GRUParam param, MatrixType type, std::vector<Shape> &in, std::vector<Shape> out) {
        Operator *op = nullptr;
        TYPE_SWITCH(type, DType, {
            op = new GRUOp<DType, cpu>(param);
        })
        return op;
    }

    GRUParam::GRUParam(MatrixType matrixType) : Parameter(matrixType) {

    }
}