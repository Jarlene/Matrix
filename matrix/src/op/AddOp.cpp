//
// Created by Jarlene on 2017/8/2.
//

#include "matrix/include/op/AddOp.h"

namespace matrix {


    template <class T, class Context>
    AddOp<T, Context>::AddOp(matrix::AddParam &param) {
        this->in = param.in;
        this->inShape = param.inShape;
        this->out = param.out;
    }


    template <class T, class Context>
    bool AddOp<T, Context>::Run() {
        int size = inShape.at(INPUT1).size();
        Add(size,  in.at(INPUT1). template Get<T>(), in.at(INPUT2). template Get<T>(), out. template GetMutable<T>());
        return true;

    }

    template <class T, class Context>
    void AddOp<T, Context>::AsyncRun() {
        if (context.mode == RunMode::kCpu) {
            Run();
        } else if (context.mode == RunMode::kGpu) {
            RunOnDevice();
        }
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