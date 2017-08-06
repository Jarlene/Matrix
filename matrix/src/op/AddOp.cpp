//
// Created by Jarlene on 2017/8/2.
//

#include "matrix/include/op/AddOp.h"

namespace matrix {


    template <class T, class Context>
    AddOp<T, Context>::AddOp(matrix::AddParam &param) {
        this->inShape.reShape(param.inShape);
        this->outShape.reShape(param.outShape);
        this->input.insert(input.end(), param.in.begin(), param.in.end());
        this->output.push_back(param.out);
        InferShape();
    }


    template <class T, class Context>
    bool AddOp<T, Context>::Run() {
        Tensor<T> t1 = Inputs()[INPUT1]. template GeneratorTensor<T>(inShape);
        Tensor<T> t2 = Inputs()[INPUT2]. template GeneratorTensor<T>(inShape);
        Tensor<T> out = Outputs()[OUT]-> template GeneratorTensor<T>(outShape);
        Add(t1, t2, out);
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

    template <class T, class Context>
    bool AddOp<T, Context>::InferShape() {
        this->outShape.reShape(this->inShape);
        return false;
    }



    template <>
    Operator* CreateOp<cpu>(AddParam &param) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new AddOp<DType, cpu>(param);
        })
        return op;
    }
}