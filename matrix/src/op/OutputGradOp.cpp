//
// Created by Jarlene on 2017/11/12.
//

#include "matrix/include/op/OutputGradOp.h"

namespace matrix {


    template <class T, class Context>
    OutputGradOp<T, Context>::OutputGradOp(Parameter &param) {
        INIT_PARAMS
    }

    template <class T, class Context>
    bool OutputGradOp<T, Context>::Run() {


        return true;
    }

    template <class T, class Context>
    void OutputGradOp<T, Context>::AsyncRun() {
        if (Context::mode == RunMode::kCpu) {
            Run();
        } else {
            if (!RunOnDevice()) {
                Run();
            }
        }
    }

    template <class T, class Context>
    bool OutputGradOp<T, Context>::RunOnDevice() {
        return false;
    }

    template <class T, class Context>
    OutputGradOp<T, Context>::~OutputGradOp() {

    }



    template <>
    Operator* CreateOp<CPU>(Parameter &param, long *size) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new OutputGradOp<DType, CPU>(param);
            *size = sizeof(DType) * param.outShapes->Size();
        })
        return op;
    }

    template <>
    Operator* CreateOp<GPU>(Parameter &param, long *size) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new OutputGradOp<DType, GPU>(param);
            *size = sizeof(DType) * param.outShapes->Size();
        })
        return op;
    }



    OutputGradOpProp::OutputGradOpProp() {
        param = new Parameter(kFloat);
    }

    OutputGradOpProp::OutputGradOpProp(const MatrixType &type) {
        param = new Parameter(type);
    }

    OutputGradOpProp::~OutputGradOpProp() {
        delete param;
    }

    void OutputGradOpProp::InferShape(std::vector<Shape*> &inShape, Shape *outShape) {
        assert(outShape != nullptr);
        outShape->reShape(*inShape[0]);
    }

    Operator *OutputGradOpProp::CreateOperator(Context context, std::vector<Blob*> &input, Blob* output,
                                           std::vector<Shape*> &inShape, Shape *outShape,
                                           std::map<std::string, Any> &args) {
        param->args = &args;
        param->inputs = input;
        param->outputs = output;
        InferShape(inShape, outShape);
        param->inputShapes = inShape;
        param->outShapes = outShape;
        BIND_DISPATCH(CreateOp, *param, &memorySize);
    }

}