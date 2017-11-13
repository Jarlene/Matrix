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

        if (HasArg("isTrain")) {
            Tensor<T> out(Output<T>(), *outputShape);
            if (HasArg("constant")) {
                T val = GetArgValue<T>("constant", T(0.1));
                Value<T>(out, val);
            } else {
                T mu  = GetArgValue<T>("mu", T(0));
                T sigma = GetArgValue<T>("sigma", T(1));
                Random<T>(out, mu, sigma);
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



    VariableOpProp::VariableOpProp() {
        param = new Parameter(kFloat);
    }

    VariableOpProp::VariableOpProp(const MatrixType &type) {
        param = new Parameter(type);
    }

    VariableOpProp::~VariableOpProp() {
        delete param;
    }

    void VariableOpProp::InferShape(std::vector<Shape*> &inShape, Shape* outShape) {
        if (outShape == nullptr) {
            Logger::Global()->Fatal("variable output shape must not null \n");
        }
    }

    Operator *VariableOpProp::CreateOperator(Context context, std::vector<void *> &input, void *output,
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
        CREATE_OPERATOR(context, param, VariableOp, {
            memorySize = sizeof(DType) * param->outShape->Size();
        })
    }

}
