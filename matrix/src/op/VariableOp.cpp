//
// Created by Jarlene on 2017/8/9.
//

#include "matrix/include/op/VariableOp.h"

namespace matrix {

    template <class T, class xpu>
    VariableOp<T, xpu>::VariableOp(Parameter &param) {
        INIT_PARAMS
    }


    template <class T, class xpu>
    bool VariableOp<T, xpu>::Run() {
        if (!init) {
            if (HasArg("isTrain")) {
                Tensor<T> out(Output<T>(), *outputShape);
                if (HasArg("constant")) {
                    T val = GetArgValue<T>("constant", T(0.1));
                    Value<T>(out, val);
                } else {
                    T mu  = GetArgValue<T>("mu", T(0));
                    T sigma = GetArgValue<T>("sigma", T(0.1));
                    Random<T>(out, mu, sigma);
                }
                init = true;
            }
        }
        return true;
    }

    template <class T, class xpu>
    void VariableOp<T, xpu>::AsyncRun() {
        if (xpu::mode == RunMode::kCpu) {
            Run();
        } else {
            if (!RunOnDevice()) {
                Run();
            }
        }
    }


    template <class T, class xpu>
    bool VariableOp<T, xpu>::RunOnDevice() {
        return false;
    }


    template <class T, class xpu>
    VariableOp<T, xpu>::~VariableOp() {

    }


    void VariableOpProp::InferShape(std::vector<Shape*> &inShape, Shape* outShape) {
        if (outShape == nullptr) {
            Logger::Global()->Fatal("variable output shape must not null \n");
        }
    }

    INIT_OPERATOR_PROPERTY_CREATE(VariableOpProp, VariableOp, true);
}
