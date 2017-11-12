//
// Created by Jarlene on 2017/8/2.
//

#include "matrix/include/op/AddOp.h"

namespace matrix {


    template <class T, class xpu>
    AddOp<T, xpu>::AddOp(Parameter &param) {
        INIT_PARAMS
    }


    template <class T, class xpu>
    bool AddOp<T, xpu>::Run() {
        if (inputShapes.size() < 2) {
            Logger::Global()->Fatal("input shape size less then 2 \n");
        }
        Tensor<T> t1 = Inputs()[INPUT1]-> template GeneratorTensor<T>(*inputShapes.at(0));
        Tensor<T> t2 = Inputs()[INPUT2]-> template GeneratorTensor<T>(*inputShapes.at(1));
        Tensor<T> out = Outputs()-> template GeneratorTensor<T>(*outputShapes);
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
    Operator* CreateOp<CPU>(Parameter &param, long *size) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new AddOp<DType, CPU>(param);
            *size = sizeof(DType) * param.outShapes->Size();
        })
        return op;
    }

    template <>
    Operator* CreateOp<GPU>(Parameter &param, long *size) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new AddOp<DType, GPU>(param);
            *size = sizeof(DType) * param.outShapes->Size();
        })
        return op;
    }

    AddOpProp::AddOpProp(const MatrixType &type)  {
        param = new Parameter(type);
    }

    AddOpProp::AddOpProp() {
        param = new Parameter(MatrixType::kFloat);
    }

    Operator *AddOpProp::CreateOperator(Context context, std::vector<Blob*> &input, Blob* output,
                                        std::vector<Shape*> &inShape, Shape *outShape,
                                        std::map<std::string, Any> &args) {
        param->args = &args;
        param->inputs = input;
        param->outputs = output;
        InferShape(inShape, outShape);
        param->inputShapes = inShape;
        param->outShapes = outShape;
        BIND_DISPATCH(CreateOp, *param, &memorySize);
    }

    void AddOpProp::InferShape(std::vector<Shape*> &inShape, Shape *outShape) {
        outShape->reShape(*inShape.at(0));
    }

    AddOpProp::~AddOpProp() {
        delete param;
    }



}