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
            Tensor<T> out = Outputs()-> template GeneratorTensor<T>(outputShapes);
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

    template <>
    Operator* CreateOp<CPU>(VariableParam &param, long *size) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new VariableOp<DType, CPU>(param);
            *size = sizeof(DType) * param.outShapes->Size();
        })
        return op;
    }

    template <>
    Operator* CreateOp<GPU>(VariableParam &param, long *size) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new VariableOp<DType, GPU>(param);
            *size = sizeof(DType) * param.outShapes->Size();
        })
        return op;
    }



    VariableOpProp::VariableOpProp() {
        param = new VariableParam(kFloat);
    }

    VariableOpProp::VariableOpProp(const MatrixType &type) {
        param = new VariableParam(type);
    }

    VariableOpProp::~VariableOpProp() {
        delete param;
    }

    void VariableOpProp::InferShape(std::vector<Shape*> &inShape, Shape* outShape) {
        if (outShape == nullptr) {
            Logger::Global()->Fatal("variable output shape must not null \n");
        }
    }

    Operator *VariableOpProp::CreateOperator(Context context, std::vector<Blob*> &input, Blob* output,
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
    void VariableOpProp::SwitchType(const MatrixType &type) {
        param->type = type;
    }
}
