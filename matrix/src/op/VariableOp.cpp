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
            Tensor<T> out = output.at(0). template GeneratorTensor<T>(outputShapes.at(0));
            if (HasArg("constant")) {
                T val = GetArgValue<T>("constant", T(0.1));
                Value<T>(out, val);
            } else {
                T mu  = GetArgValue<T>("mu", T(0));
                T sigma = GetArgValue<T>("sigma", T(1));;
                Random<T>(out, mu, sigma);
            }
        }
        return true;
    }

    template <class T, class xpu>
    void VariableOp<T, xpu>::AsyncRun() {

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
            int shape = 0;
            for (Shape s : param.outShapes) {
                shape += s.Size();
            }
            *size = sizeof(DType) * shape;
        })
        return op;
    }

    template <>
    Operator* CreateOp<GPU>(VariableParam &param, long *size) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new VariableOp<DType, GPU>(param);
            int shape = 0;
            for (Shape s : param.outShapes) {
                shape += s.Size();
            }
            *size = sizeof(DType) * shape;
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

    void VariableOpProp::InferShape(std::vector<Shape> &inShape, std::vector<Shape> &outShape) {
        if (outShape.size() == 0) {
            Logger::Global()->Fatal("variable input shapes must lager then 0\n");
        }
    }

    Operator *VariableOpProp::CreateOperator(Context context, std::vector<Blob> &input, std::vector<Blob> &output,
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
