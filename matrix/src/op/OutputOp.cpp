//
// Created by Jarlene on 2017/8/9.
//

#include "matrix/include/op/OutputOp.h"

namespace matrix {
    template <class T, class Context>
    OutputOp<T, Context>::OutputOp(OutputParam &param) {

    }

    template <class T, class Context>
    bool OutputOp<T, Context>::Run() {
        return OutputOp::Run();
    }

    template <class T, class Context>
    void OutputOp<T, Context>::AsyncRun() {
        Operator::AsyncRun();
    }

    template <class T, class Context>
    bool OutputOp<T, Context>::RunOnDevice() {
        return false;
    }

    template <class T, class Context>
    OutputOp<T, Context>::~OutputOp() {

    }



    template <>
    Operator* CreateOp<CPU>(OutputParam &param, long *size) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new OutputOp<DType, CPU>(param);
            int shape = 0;
            for (Shape s : param.outShapes) {
                shape += s.Size();
            }
            *size = sizeof(DType) * shape;
        })
        return op;
    }

    template <>
    Operator* CreateOp<GPU>(OutputParam &param, long *size) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new OutputOp<DType, GPU>(param);
            int shape = 0;
            for (Shape s : param.outShapes) {
                shape += s.Size();
            }
            *size = sizeof(DType) * shape;
        })
        return op;
    }



    OutputOpProp::OutputOpProp() {
        param = new OutputParam(kFloat);
    }

    OutputOpProp::OutputOpProp(const MatrixType &type) {
        param = new OutputParam(type);
    }

    OutputOpProp::~OutputOpProp() {
        delete param;
    }

    void OutputOpProp::InferShape(std::vector<Shape> &inShape, std::vector<Shape> &outShape) {

    }

    Operator *OutputOpProp::CreateOperator(Context context, std::vector<Blob> &input, std::vector<Blob> &output,
                                               std::vector<Shape> &inShape, std::vector<Shape> &outShape,
                                               std::map<std::string, Any> &args) {
        InferShape(inShape, outShape);
        param->outputs = output;
        param->inputs = input;
        param->inputShapes = inShape;
        param->outShapes = outShape;
        param->args = args;
        BIND_DISPATCH(CreateOp, *param, &memorySize);
    }

}