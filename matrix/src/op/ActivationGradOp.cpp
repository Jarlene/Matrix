//
// Created by Jarlene on 2017/8/23.
//

#include "matrix/include/op/ActivationGradOp.h"

namespace matrix {

    template <class T, class Context>
    ActivationGradOp<T, Context>::ActivationGradOp(Parameter &param) {
        INIT_PARAMS
    }

    template <class T, class Context>
    bool ActivationGradOp<T, Context>::Run() {
        auto type = ActType::kSigmoid;
        if (HasArg("type")) {
            type = GetArgValue<ActType>("type");
        }

        Tensor<T> pre = Inputs()[PRE_GRAD]. template GeneratorTensor<T>(inputShapes[PRE_GRAD]);
        Tensor<T> out = Inputs()[OUT]. template GeneratorTensor<T>(inputShapes[OUT]);
        Tensor<T> input = Inputs()[INPUT]. template GeneratorTensor<T>(inputShapes[INPUT]);
        Tensor<T> gradOut = Outputs()[GRAD_OUT]. template GeneratorTensor<T>(outputShapes[GRAD_OUT]);

        switch (type) {
            case kSigmoid:
                SigmoidGrad<T>(input, pre, gradOut);
                break;
            case kTanh:
                TanhGrad<T>(input, pre, gradOut);
                break;
            case kRelu:
                ReluGrad<T>(input, pre, out);
                break;
            default:
                Logger::Global()->Fatal("ActivationOp not support \n");
                break;
        }
        return true;
    }

    template <class T, class Context>
    void ActivationGradOp<T, Context>::AsyncRun() {
        if (Context::mode == RunMode::kCpu) {
            Run();
        } else if (Context::mode == RunMode::kGpu){
            if (!RunOnDevice()) {
                Run();
            }
        }
    }

    template <class T, class Context>
    bool ActivationGradOp<T, Context>::RunOnDevice() {
        return false;
    }

    template <class T, class Context>
    ActivationGradOp<T, Context>::~ActivationGradOp() {

    }



    ActivationOpGradProp::ActivationOpGradProp() {
        param = new ActivationGradParam(kFloat);
    }

    ActivationOpGradProp::ActivationOpGradProp(const MatrixType &type) {
        param = new ActivationGradParam(type);
    }

    ActivationOpGradProp::~ActivationOpGradProp() {
        delete param;
    }

    void ActivationOpGradProp::InferShape(std::vector<Shape> &inShape, std::vector<Shape> &outShape) {
        assert(inShape.size() == 3);
        outShape.at(0).reShape(inShape[2]);
    }


    template <>
    Operator* CreateOp<CPU>(ActivationGradParam &param, long *size) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new ActivationGradOp<DType, CPU>(param);
            int shape = 0;
            for (Shape s : param.outShapes) {
                shape += s.Size();
            }
            *size = sizeof(DType) * shape;
        })
        return op;
    }

    template <>
    Operator* CreateOp<GPU>(ActivationGradParam &param, long *size) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new ActivationGradOp<DType, GPU>(param);
            int shape = 0;
            for (Shape s : param.outShapes) {
                shape += s.Size();
            }
            *size = sizeof(DType) * shape;
        })
        return op;
    }


    Operator *ActivationOpGradProp::CreateOperator(Context context, std::vector<Blob> &input, std::vector<Blob> &output,
                                                   std::vector<Shape> &inShape, std::vector<Shape> &outShape,
                                                   std::map<std::string, Any> &args) {
        param->args = args;
        param->inputs = input;
        param->outputs = output;
        InferShape(inShape, outShape);
        param->inputShapes = inShape;
        param->outShapes = outShape;
        BIND_DISPATCH(CreateOp, *param, &memorySize);
    }


}