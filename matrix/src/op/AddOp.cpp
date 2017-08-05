//
// Created by Jarlene on 2017/8/2.
//

#include "matrix/include/op/AddOp.h"

namespace matrix {


    template <class T, class Context>
    AddOp<T, Context>::AddOp(matrix::AddParam &param) : inShape(param.inShape) , outShape(param.outShape){
        this->input.insert(input.end(), param.in.end(), param.in.end());
        this->output.push_back(param.out);
        InferShape();
    }


    template <class T, class Context>
    bool AddOp<T, Context>::Run() {
        int size = inShape.Size();
        Add(size,  Inputs().at(INPUT1). template Get<T>(), Inputs().at(INPUT2). template Get<T>(), output.at(0)-> template GetMutable<T>());
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