//
// Created by Jarlene on 2017/8/2.
//

#include "matrix/include/op/ActivationOp.h"

namespace matrix {


    template <class T, class Context>
    ActivationOp<T, Context>::ActivationOp(Parameter &param) {
        INIT_PARAMS
    }

    template <class T, class Context>
    bool ActivationOp<T, Context>::Run() {
        auto type = ActType::kSigmoid;
        if (HasArg("type")) {
            type = GetArgValue<ActType>("type");
        }
        Tensor<T> data = Inputs()[DATA]-> template GeneratorTensor<T>(*inputShapes[DATA]);
        Tensor<T> out = Outputs()-> template GeneratorTensor<T>(*outputShapes);
        switch (type) {
            case kSigmoid:
                Sigmoid<T>(data, out);
                break;
            case kTanh:
                Tanh<T>(data, out);
                break;
            case kRelu:
                Relu<T>(data, out);
                break;
            default:
                Logger::Global()->Fatal("ActivationOp not support \n");
                break;
        }
        return true;
    }

    template <class T, class Context>
    void ActivationOp<T, Context>::AsyncRun() {
        if (Context::mode == RunMode::kCpu) {
            Run();
        } else if (Context::mode == RunMode::kGpu){
            if (!RunOnDevice()) {
                Run();
            }
        }
    }

    template <class T, class Context>
    bool ActivationOp<T, Context>::RunOnDevice() {
        return false;
    }

    template <class T, class Context>
    ActivationOp<T, Context>::~ActivationOp() {

    }


    template <>
    Operator* CreateOp<CPU>(ActivationParam &param, long *size) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new ActivationOp<DType, CPU>(param);
            *size = sizeof(DType) * param.outShapes->Size();
        })
        return op;
    }

    template <>
    Operator* CreateOp<GPU>(ActivationParam &param, long *size) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new ActivationOp<DType, GPU>(param);
            *size = sizeof(DType) * param.outShapes->Size();
        })
        return op;
    }




    ActivationOpProp::ActivationOpProp() {
        param = new ActivationParam(kFloat);
    }

    ActivationOpProp::ActivationOpProp(const MatrixType &type) {
        param = new ActivationParam(type);
    }

    ActivationOpProp::~ActivationOpProp() {
        delete param;
    }

    void ActivationOpProp::InferShape(std::vector<Shape*> &inShape, Shape* outShape) {
        outShape->reShape(*inShape[0]);
    }

    Operator *ActivationOpProp::CreateOperator(Context context, std::vector<Blob*> &input, Blob* output,
                                               std::vector<Shape*> &inShape, Shape* outShape,
                                               std::map<std::string, Any> &args) {
        param->args = &args;
        param->inputs = input;
        param->outputs = output;
        InferShape(inShape, outShape);
        param->inputShapes = inShape;
        param->outShapes = outShape;
        BIND_DISPATCH(CreateOp, *param, &memorySize);
    }

    void ActivationOpProp::SwitchType(const MatrixType &type) {
        this->param->type = type;
    }
}