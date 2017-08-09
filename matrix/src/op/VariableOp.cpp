//
// Created by Jarlene on 2017/8/9.
//

#include "matrix/include/op/VariableOp.h"

namespace matrix {

    template <class T, class xpu>
    VariableOp<T, xpu>::VariableOp(VariableParam &param) {

    }


    template <class T, class xpu>
    bool VariableOp<T, xpu>::Run() {
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
    Operator* CreateOp<CPU>(VariableParam &param) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new VariableOp<DType, CPU>(param);
        })
        return op;
    }

    template <>
    Operator* CreateOp<GPU>(VariableParam &param) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new VariableOp<DType, GPU>(param);
        })
        return op;
    }



    VariableOpProp::VariableOpProp() {
        param = new VariableParam(kFloat);
    }

    VariableOpProp::VariableOpProp(const MatrixType type) {
        param = new VariableParam(type);
    }

    VariableOpProp::~VariableOpProp() {
        delete param;
    }

    void VariableOpProp::InferShape(std::vector<Shape> &inShape, std::vector<Shape> &outShape) {

    }

    Operator *VariableOpProp::CreateOperator(Context context, std::vector<Blob> &input, std::vector<Blob> &output,
                                                std::vector<Shape> &inShape, std::vector<Shape> &outShape) {
        InferShape(inShape, outShape);
        param->outputs = output;
        param->inputs = input;
        param->inputShapes = inShape;
        param->outShapes = outShape;
        BIND_DISPATCH(CreateOp, *param);
    }
}
