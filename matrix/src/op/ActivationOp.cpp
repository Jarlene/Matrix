//
// Created by Jarlene on 2017/8/2.
//

#include "matrix/include/op/ActivationOp.h"

namespace matrix {


    template <class T, class xpu>
    ActivationOp<T, xpu>::ActivationOp(Parameter &param) {
        INIT_PARAMS
    }

    template <class T, class xpu>
    bool ActivationOp<T, xpu>::Run() {
        auto type = GetArgValue<ActType>("type", kSigmoid);
        Tensor<T> data(Input<T>(DATA) , *inputShapes->at(DATA));
        Tensor<T> out(Output<T>(), *outputShape);
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

    template <class T, class xpu>
    void ActivationOp<T, xpu>::AsyncRun() {
        if (xpu::mode == RunMode::kCpu) {
            Run();
        } else if (xpu::mode == RunMode::kGpu){
            if (!RunOnDevice()) {
                Run();
            }
        }
    }

    template <class T, class xpu>
    bool ActivationOp<T, xpu>::RunOnDevice() {
        return false;
    }

    template <class T, class xpu>
    ActivationOp<T, xpu>::~ActivationOp() {

    }



    void ActivationOpProp::InferShape(std::vector<Shape*> &inShape, Shape* outShape) {
        outShape->reShape(*inShape[0]);
    }


    INIT_OPERATOR_PROPERTY_CREATE(ActivationOpProp, ActivationOp, true);


}