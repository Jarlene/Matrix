//
// Created by Jarlene on 2017/11/5.
//

#include "matrix/include/op/Convolution1DGradOp.h"


namespace matrix {

    template<class T, class Context>
    Convolution1DGradOp<T, Context>::Convolution1DGradOp(Parameter &param) {
        INIT_PARAMS

    }

    template<class T, class Context>
    bool Convolution1DGradOp<T, Context>::Run() {
        return true;
    }


    template<class T, class Context>
    void Convolution1DGradOp<T, Context>::AsyncRun() {
        if (Context::mode == RunMode::kCpu) {
            Run();
        } else {
            if (!RunOnDevice()) {
                Run();
            }
        }
    }

    template<class T, class Context>
    Convolution1DGradOp<T, Context>::~Convolution1DGradOp() {

    }

    template<class T, class Context>
    bool Convolution1DGradOp<T, Context>::RunOnDevice() {
        return false;
    }


    template<>
    Operator *CreateOp<CPU>(Parameter &param, long *size) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new Convolution1DGradOp<DType, CPU>(param);
            *size = sizeof(DType) * param.outShapes->Size();
        })
        return op;
    }

    template<>
    Operator *CreateOp<GPU>(Parameter &param, long *size) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new Convolution1DGradOp<DType, GPU>(param);
            *size = sizeof(DType) * param.outShapes->Size();
        })
        return op;
    }


    Convolution1DGradOpProp::Convolution1DGradOpProp() {
        param = new Parameter(kFloat);
    }

    Convolution1DGradOpProp::Convolution1DGradOpProp(const MatrixType &type) {
        param = new Parameter(type);
    }

    Convolution1DGradOpProp::~Convolution1DGradOpProp() {
        delete param;

    }

    void Convolution1DGradOpProp::InferShape(std::vector<Shape *> &inShape, Shape *outShape) {

    }

    Operator *Convolution1DGradOpProp::CreateOperator(Context context, std::vector<Blob *> &input, Blob *output,
                                                      std::vector<Shape *> &inShape, Shape *outShape,
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