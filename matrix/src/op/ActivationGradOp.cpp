//
// Created by Jarlene on 2017/8/23.
//

#include "matrix/include/op/ActivationGradOp.h"

namespace matrix {

    template <class T, class xpu>
    ActivationGradOp<T, xpu>::ActivationGradOp(Parameter &param) {
        INIT_PARAMS

    }

    template <class T, class xpu>
    bool ActivationGradOp<T, xpu>::Run() {
        auto type = ActType::kSigmoid;
        if (HasArg("type")) {
            type = GetArgValue<ActType>("type");
        }

        Tensor<T> pre(Input<T>(PRE_GRAD),*inputShapes->at(PRE_GRAD));
        Tensor<T> out (Input<T>(OUT), *inputShapes->at(OUT));
        Tensor<T> input(Input<T>(INPUT), *inputShapes->at(INPUT));
        Tensor<T> gradOut(Output<T>() ,*outputShape);

        switch (type) {
            case kSigmoid:
                SigmoidGrad<T>(out, pre, gradOut);
                break;
            case kTanh:
                TanhGrad<T>(out, pre, gradOut);
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

    template <class T, class xpu>
    void ActivationGradOp<T, xpu>::AsyncRun() {
        if (xpu::mode == RunMode::kCpu) {
            Run();
        } else if (xpu::mode == RunMode::kGpu){
            if (!RunOnDevice()) {
                Run();
            }
        }
    }

    template <class T, class xpu>
    bool ActivationGradOp<T, xpu>::RunOnDevice() {
        return false;
    }

    template <class T, class xpu>
    ActivationGradOp<T, xpu>::~ActivationGradOp() {

    }


    void ActivationOpGradProp::InferShape(std::vector<Shape*> &inShape, Shape* outShape) {
        assert(outShape != nullptr);
        outShape->reShape(*inShape[2]);
    }


    INIT_OPERATOR_PROPERTY_CREATE(ActivationOpGradProp, ActivationGradOp, true);



}