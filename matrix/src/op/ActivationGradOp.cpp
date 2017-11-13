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

        Tensor<T> pre(Input<T>(PRE_GRAD),*inputShapes[PRE_GRAD]);
        Tensor<T> out (Input<T>(OUT), *inputShapes[OUT]);
        Tensor<T> input(Input<T>(INPUT), *inputShapes[INPUT]);
        Tensor<T> gradOut(Output<T>() ,*outputShape);

        switch (type) {
            case kSigmoid:
                SigmoidGrad<T>(out, gradOut);
                break;
            case kTanh:
                TanhGrad<T>(out, gradOut);
                break;
            case kRelu:
                ReluGrad<T>(input, pre, gradOut);
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
        param = new Parameter(kFloat);
    }

    ActivationOpGradProp::ActivationOpGradProp(const MatrixType &type) {
        param = new Parameter(type);
    }

    ActivationOpGradProp::~ActivationOpGradProp() {
        delete param;
    }

    void ActivationOpGradProp::InferShape(std::vector<Shape*> &inShape, Shape* outShape) {
        assert(outShape != nullptr);
        outShape->reShape(*inShape[2]);
    }




    Operator *ActivationOpGradProp::CreateOperator(Context context, std::vector<void *> &input, void *output,
                                                   std::vector<Shape *> &inShape, Shape *outShape,
                                                   std::map<std::string, Any> &args) {
        param->args = &args;
        param->output = output;
        InferShape(inShape, outShape);
        for(auto it = inShape.begin(); it != inShape.end(); ++it) {
            param->inputShapes.push_back(*it);
        }
        for(auto it = input.begin(); it != input.end(); ++it) {
            param->inputs.push_back(*it);
        }
        param->outShape = outShape;
        CREATE_OPERATOR(context, param, ActivationGradOp, {
            memorySize = sizeof(DType) * param->outShape->Size();
        })
    }



}