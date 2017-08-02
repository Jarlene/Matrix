//
// Created by Jarlene on 2017/8/2.
//

#include "matrix/include/op/AddOp.h"

namespace matrix {


    template <class T, class Context>
    AddOp<T, Context>::AddOp(matrix::AddParam &param) {

    }


    template <class T, class Context>
    bool AddOp<T, Context>::Run() {
        return Operator::Run();
    }

    template <class T, class Context>
    void AddOp<T, Context>::AsyncRun() {
        Operator::AsyncRun();
    }

    template <class T, class Context>
    AddOp<T, Context>::~AddOp() {

    }

    template <class T, class Context>
    bool AddOp<T, Context>::RunOnDevice() {
        return false;
    }



    template <>
    Operator* CreateOp<cpu>(AddParam param, MatrixType type, std::vector<Shape> &in, std::vector<Shape> out) {
        Operator *op = nullptr;
        TYPE_SWITCH(type, DType, {
            op = new AddOp<DType, cpu>(param);
        })
        return op;
    }
}