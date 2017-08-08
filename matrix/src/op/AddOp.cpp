//
// Created by Jarlene on 2017/8/2.
//

#include "matrix/include/op/AddOp.h"

namespace matrix {


    template <class T, class xpu>
    AddOp<T, xpu>::AddOp(matrix::AddParam &param) {
        this->args["input_shape"] = param.inShapes;
        this->args["output_shape"] = param.outShape;
        this->input.insert(input.end(), param.in.begin(), param.in.end());
        this->output.push_back(param.out);
    }


    template <class T, class xpu>
    bool AddOp<T, xpu>::Run() {
        std::vector<Shape> inShapes = GetArgValue<std::vector<Shape>>("input_shape");
        if (inShapes.size() < 2) {
            Logger::Global()->Fatal("input shape size less then 2 \n");
        }
        Shape outShape = GetArgValue<Shape>("output_shape");
        Tensor<T> t1 = Inputs()[INPUT1]. template GeneratorTensor<T>(inShapes.at(0));
        Tensor<T> t2 = Inputs()[INPUT2]. template GeneratorTensor<T>(inShapes.at(1));
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

    AddOpProp::AddOpProp(const MatrixType type)  {
        param = new AddParam(type);
    }

    AddOpProp::AddOpProp() {
        param = new AddParam(MatrixType::kFloat);
    }

    Operator *AddOpProp::CreateOperator(Context context, std::vector<Blob> &input, std::vector<Blob> &output, std::vector<Shape> &inShape, std::vector<Shape> &outShape) {
        InferShape(inShape, outShape);
        param->out = &output.at(0);
        param->in = input;
        param->inShapes = inShape;
        param->outShape = outShape.at(0);
        BIND_DISPATCH(CreateOp, *param);
    }

    void AddOpProp::InferShape(std::vector<Shape> &inShape, std::vector<Shape> &outShape) {
        outShape.at(0).reShape(inShape.at(0));
    }

    AddOpProp::~AddOpProp() {
        delete param;
    }


}