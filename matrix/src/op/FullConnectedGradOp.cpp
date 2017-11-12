//
// Created by Jarlene on 2017/8/23.
//


#include "matrix/include/op/FullConnectedGradOp.h"

namespace matrix {

    template <class T, class Context>
    FullConnectedGradOp<T, Context>::FullConnectedGradOp(Parameter &param) {
        INIT_PARAMS
    }

    template <class T, class Context>
    bool FullConnectedGradOp<T, Context>::Run() {
        if (!HasArg("input_idx")) {
            Logger::Global()->Fatal("FullConnectedGradOp not support. \n");
        }
        int idx = GetArgValue<int>("input_idx");
        Tensor<T> pre_grad = Inputs()[PRE_GRAD]-> template GeneratorTensor<T>(inputShapes[PRE_GRAD]);
        Tensor<T> data = Inputs()[DATA]-> template GeneratorTensor<T>(inputShapes[DATA]);
        Tensor<T> weight = Inputs()[WEIGHT]-> template GeneratorTensor<T>(inputShapes[WEIGHT]);

        Tensor<T> out = Outputs()-> template GeneratorTensor<T>(outputShapes);
        switch (idx + 2) {
            case DATA:
                MatrixMul<T>(pre_grad, false, weight, true, out);
                break;
            case WEIGHT:
                MatrixMul<T>(data, true, pre_grad, false, out);
                break;
            case BIAS:
                Copy<T>(pre_grad, out);
                break;
            default:
                Logger::Global()->Fatal("FullConnectedGradOp not support. \n");
                break;
        }
        return true;
    }

    template <class T, class Context>
    void FullConnectedGradOp<T, Context>::AsyncRun() {
        if (Context::mode == RunMode::kCpu) {
            Run();
        } else {
            if (!RunOnDevice()) {
                Run();
            }
        }
    }

    template <class T, class Context>
    FullConnectedGradOp<T, Context>::~FullConnectedGradOp() {

    }

    template <class T, class Context>
    bool FullConnectedGradOp<T, Context>::RunOnDevice() {
        return false;
    }




    template <>
    Operator* CreateOp<CPU>(Parameter &param, long *size) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new FullConnectedGradOp<DType, CPU>(param);
            *size = sizeof(DType) * param.outShapes->Size();
        })
        return op;
    }

    template <>
    Operator* CreateOp<GPU>(Parameter &param, long *size) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new FullConnectedGradOp<DType, GPU>(param);
            *size = sizeof(DType) * param.outShapes->Size();
        })
        return op;
    }


    FullConnectedGradOpProp::FullConnectedGradOpProp() {
        param = new Parameter(kFloat);
    }

    FullConnectedGradOpProp::FullConnectedGradOpProp(const MatrixType &type) {
        param = new Parameter(type);
    }

    FullConnectedGradOpProp::~FullConnectedGradOpProp() {
        delete param;
    }

    void FullConnectedGradOpProp::InferShape(std::vector<Shape*> &inShape, Shape *outShape) {
        if (param->args->count("input_idx")) {
            int idx = get<int>(param->args->at("input_idx"));
            outShape->reShape(*inShape.at(idx + 2));
        }

    }

    Operator *FullConnectedGradOpProp::CreateOperator(Context context, std::vector<Blob*> &input, Blob* output,
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



}