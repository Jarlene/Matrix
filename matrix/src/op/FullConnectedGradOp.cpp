//
// Created by Jarlene on 2017/8/23.
//


#include "matrix/include/op/ReduceOp.h"
#include "matrix/include/op/FullConnectedGradOp.h"

namespace matrix {

    template <class T, class xpu>
    FullConnectedGradOp<T, xpu>::FullConnectedGradOp(Parameter &param) {
        INIT_PARAMS
    }

    template <class T, class xpu>
    bool FullConnectedGradOp<T, xpu>::Run() {
        if (!HasArg("input_idx")) {
            Logger::Global()->Fatal("FullConnectedGradOp not support. \n");
        }
        int idx = GetArgValue<int>("input_idx");
        Tensor<T> pre_grad(Input<T>(PRE_GRAD), *inputShapes->at(PRE_GRAD));
        Tensor<T> self_out(Input<T>(OUT), *inputShapes->at(OUT));
        Tensor<T> data(Input<T>(DATA), *inputShapes->at(DATA));
        Tensor<T> weight(Input<T>(WEIGHT), *inputShapes->at(WEIGHT));

        Tensor<T> out(Output<T>(), *outputShape);
        if (HasArg("activation_type")) {
            auto actType = GetArgValue<ActType>("activation_type");
            switch (actType) {
                case kSigmoid:
                    SigmoidGrad<T>(self_out, pre_grad, pre_grad);
                    break;
                case kTanh:
                    TanhGrad<T>(self_out, pre_grad, pre_grad);
                    break;
                case kRelu:
                    ReluGrad<T>(self_out, pre_grad, pre_grad);
                    break;
                default:
                    Logger::Global()->Fatal("FullConnectedGradOp activation_type not support \n");
                    break;
            }
        }
        switch (idx + 2) {
            case DATA:
                MatrixMul<T>(pre_grad, false, weight, true, out);
                break;
            case WEIGHT:
                MatrixMul<T>(data, true, pre_grad, false, out);
                break;
            case BIAS:
                Sum(pre_grad, 0, out);
                break;
            default:
                Logger::Global()->Fatal("FullConnectedGradOp not support. \n");
                break;
        }
        return true;
    }

    template <class T, class xpu>
    void FullConnectedGradOp<T, xpu>::AsyncRun() {
        if (xpu::mode == RunMode::kCpu) {
            Run();
        } else {
            if (!RunOnDevice()) {
                Run();
            }
        }
    }

    template <class T, class xpu>
    FullConnectedGradOp<T, xpu>::~FullConnectedGradOp() {

    }

    template <class T, class xpu>
    bool FullConnectedGradOp<T, xpu>::RunOnDevice() {
        return false;
    }




    void FullConnectedGradOpProp::InferShape(std::vector<Shape*> &inShape, Shape *outShape) {
        if (param->args->count("input_idx")) {
            int idx = get<int>(param->args->at("input_idx"));
            outShape->reShape(*inShape.at(idx + 2));
        }

    }
    INIT_OPERATOR_PROPERTY_CREATE(FullConnectedGradOpProp, FullConnectedGradOp, true);


}