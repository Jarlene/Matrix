//
// Created by Jarlene on 2017/8/2.
//

#include "matrix/include/op/AddOp.h"

namespace matrix {


    template <class T, class xpu>
    AddOp<T, xpu>::AddOp(Parameter &param) {
        this->inputShapes = param.inputShapes;
        this->outputShapes = param.outShapes;
        this->input.insert(input.end(), param.inputs.begin(), param.inputs.end());
        for (Blob b : param.outputs) {
            this->output.push_back(b);
        }
    }


    template <class T, class xpu>
    bool AddOp<T, xpu>::Run() {
        if (inputShapes.size() < 2) {
            Logger::Global()->Fatal("input shape size less then 2 \n");
        }
        Tensor<T> t1 = Inputs()[INPUT1]. template GeneratorTensor<T>(inputShapes.at(0));
        Tensor<T> t2 = Inputs()[INPUT2]. template GeneratorTensor<T>(inputShapes.at(1));
        Tensor<T> out = Outputs()[OUT]. template GeneratorTensor<T>(outputShapes.at(0));
        Add(t1, t2, out);
        return true;

    }

    template <class T, class xpu>
    void AddOp<T, xpu>::AsyncRun() {
        if (xpu::mode == RunMode::kCpu) {
            Run();
        } else if (xpu::mode == RunMode::kGpu) {
            if (!RunOnDevice()) {
                Run();
            }
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
    Operator* CreateOp<CPU>(AddParam &param, long *size) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new AddOp<DType, CPU>(param);
            int shape = 0;
            for (Shape s : param.outShapes) {
                shape += s.Size();
            }
            *size = sizeof(DType) * shape;
        })
        return op;
    }

    template <>
    Operator* CreateOp<GPU>(AddParam &param, long *size) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new AddOp<DType, GPU>(param);
            int shape = 0;
            for (Shape s : param.outShapes) {
                shape += s.Size();
            }
            *size = sizeof(DType) * shape;
        })
        return op;
    }

    AddOpProp::AddOpProp(const MatrixType &type)  {
        param = new AddParam(type);
    }

    AddOpProp::AddOpProp() {
        param = new AddParam(MatrixType::kFloat);
    }

    Operator *AddOpProp::CreateOperator(Context context, std::vector<Blob> &input, std::vector<Blob> &output,
                                        std::vector<Shape> &inShape, std::vector<Shape> &outShape,
                                        std::map<std::string, Any> &args) {
        InferShape(inShape, outShape);
        param->outputs = output;
        param->inputs = input;
        param->inputShapes = inShape;
        param->outShapes = outShape;
        param->args = args;
        BIND_DISPATCH(CreateOp, *param, &memorySize);
    }

    void AddOpProp::InferShape(std::vector<Shape> &inShape, std::vector<Shape> &outShape) {
        outShape.at(0).reShape(inShape.at(0));
    }

    AddOpProp::~AddOpProp() {
        delete param;
    }


}