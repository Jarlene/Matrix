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



    void ActivationOpProp::InferShape(std::vector<Shape*> &inShape, Shape* outShape) {
        outShape->reShape(*inShape[0]);
    }


    INIT_OPERATOR_PROPERTY_CREATE(ActivationOpProp, ActivationOp, true);


}