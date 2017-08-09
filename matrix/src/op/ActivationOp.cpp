//
// Created by Jarlene on 2017/8/2.
//

#include "matrix/include/op/ActivationOp.h"

namespace matrix {


    template <class T, class Context>
    ActivationOp<T, Context>::ActivationOp(matrix::ActivationParam &param) {

    }

    template <class T, class Context>
    bool ActivationOp<T, Context>::Run() {
        return Operator::Run();
    }

    template <class T, class Context>
    void ActivationOp<T, Context>::AsyncRun() {
        Operator::AsyncRun();
    }

    template <class T, class Context>
    bool ActivationOp<T, Context>::RunOnDevice() {
        return false;
    }

    template <class T, class Context>
    ActivationOp<T, Context>::~ActivationOp() {

    }


    template <>
    Operator* CreateOp<CPU>(ActivationParam &param, long *size) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new ActivationOp<DType, CPU>(param);
            int shape = 0;
            for (Shape s : param.outShapes) {
                shape += s.Size();
            }
            *size = sizeof(DType) * shape;
        })
        return op;
    }

    template <>
    Operator* CreateOp<GPU>(ActivationParam &param, long *size) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new ActivationOp<DType, GPU>(param);
            int shape = 0;
            for (Shape s : param.outShapes) {
                shape += s.Size();
            }
            *size = sizeof(DType) * shape;
        })
        return op;
    }




    ActivationOpProp::ActivationOpProp() {
        param = new ActivationParam(kFloat);
    }

    ActivationOpProp::ActivationOpProp(const MatrixType &type) {
        param = new ActivationParam(type);
    }

    ActivationOpProp::~ActivationOpProp() {
        delete param;
    }

    void ActivationOpProp::InferShape(std::vector<Shape> &inShape, std::vector<Shape> &outShape) {

    }

    Operator *ActivationOpProp::CreateOperator(Context context, std::vector<Blob> &input, std::vector<Blob> &output,
                                               std::vector<Shape> &inShape, std::vector<Shape> &outShape,
                                               std::map<std::string, Any> &args) {
        InferShape(inShape, outShape);
        param->inputs = input;
        param->outputs = output;
        param->inputShapes = inShape;
        param->outShapes = outShape;
        param->args = args;
        BIND_DISPATCH(CreateOp, *param, &memorySize);
    }
}