//
// Created by Jarlene on 2017/8/23.
//

#include "matrix/include/op/ActivationGradOp.h"

namespace matrix {

    template <class T, class Context>
    ActivationGradOp<T, Context>::ActivationGradOp(Parameter &param) {
        this->outputShapes = param.outShapes;
        this->output = param.outputs;
        this->input = param.inputs;
        this->inputShapes = param.inputShapes;
    }

    template <class T, class Context>
    bool ActivationGradOp<T, Context>::Run() {
        return Operator::Run();
    }

    template <class T, class Context>
    void ActivationGradOp<T, Context>::AsyncRun() {
        Operator::AsyncRun();
    }

    template <class T, class Context>
    bool ActivationGradOp<T, Context>::RunOnDevice() {
        return false;
    }

    template <class T, class Context>
    ActivationGradOp<T, Context>::~ActivationGradOp() {

    }



    ActivationOpGradProp::ActivationOpGradProp() {
        param = new ActivationGradParam(kFloat);
    }

    ActivationOpGradProp::ActivationOpGradProp(const MatrixType &type) {
        param = new ActivationGradParam(type);
    }

    ActivationOpGradProp::~ActivationOpGradProp() {
        delete param;
    }

    void ActivationOpGradProp::InferShape(std::vector<Shape> &inShape, std::vector<Shape> &outShape) {

    }





    template <>
    Operator* CreateOp<CPU>(ActivationGradParam &param, long *size) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new ActivationGradOp<DType, CPU>(param);
            int shape = 0;
            for (Shape s : param.outShapes) {
                shape += s.Size();
            }
            *size = sizeof(DType) * shape;
        })
        return op;
    }

    template <>
    Operator* CreateOp<GPU>(ActivationGradParam &param, long *size) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new ActivationGradOp<DType, GPU>(param);
            int shape = 0;
            for (Shape s : param.outShapes) {
                shape += s.Size();
            }
            *size = sizeof(DType) * shape;
        })
        return op;
    }


    Operator *ActivationOpGradProp::CreateOperator(Context context, std::vector<Blob> &input, std::vector<Blob> &output,
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