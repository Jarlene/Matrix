//
// Created by Jarlene on 2017/8/2.
//

#include "matrix/include/op/AddOp.h"

namespace matrix {


    template <class T, class xpu>
    AddOp<T, xpu>::AddOp(matrix::AddParam &param) {
        this->inShape.reShape(param.inShape);
        this->outShape.reShape(param.outShape);
        this->input.insert(input.end(), param.in.begin(), param.in.end());
        this->output.push_back(param.out);
        InferShape();
    }


    template <class T, class xpu>
    bool AddOp<T, xpu>::Run() {
        Tensor<T> t1 = Inputs()[INPUT1]. template GeneratorTensor<T>(inShape);
        Tensor<T> t2 = Inputs()[INPUT2]. template GeneratorTensor<T>(inShape);
        Tensor<T> out = Outputs()[OUT]-> template GeneratorTensor<T>(outShape);
        Add(t1, t2, out);
        return true;

    }

    template <class T, class xpu>
    void AddOp<T, xpu>::AsyncRun() {
        if (xpu::mode == RunMode::kCpu) {
            Run();
        } else if (xpu::mode == RunMode::kGpu) {
            RunOnDevice();
        }
    }

    template <class T, class xpu>
    AddOp<T, xpu>::~AddOp() {

    }

    template <class T, class xpu>
    bool AddOp<T, xpu>::RunOnDevice() {
        return false;
    }




    template <>
    Operator* CreateOp<CPU>(AddParam &param) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new AddOp<DType, CPU>(param);
        })
        return op;
    }

    template <>
    Operator* CreateOp<GPU>(AddParam &param) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new AddOp<DType, GPU>(param);
        })
        return op;
    }

    Operator *AddOpProp::CreateOperator(Context context, std::vector<Shape> *inShape, std::vector<Shape> *outShape) const {
        InferShape(inShape, outShape);
        BIND_DISPATCH(CreateOp, param);
    }

    void AddOpProp::InferShape(std::vector<Shape> *inShape, std::vector<Shape> *outShape) const {
        outShape->at(0).reShape(inShape->at(0));
    }
}