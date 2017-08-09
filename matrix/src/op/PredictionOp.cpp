//
// Created by Jarlene on 2017/8/9.
//

#include "matrix/include/op/PredictionOp.h"

namespace matrix {

    template <class T, class Context>
    PredictionOp<T, Context>::PredictionOp(PredictionParam &param) {

    }

    template <class T, class Context>
    bool PredictionOp<T, Context>::Run() {
        return PredictionOp::Run();
    }

    template <class T, class Context>
    void PredictionOp<T, Context>::AsyncRun() {
        Operator::AsyncRun();
    }

    template <class T, class Context>
    bool PredictionOp<T, Context>::RunOnDevice() {
        return false;
    }

    template <class T, class Context>
    PredictionOp<T, Context>::~PredictionOp() {

    }



    template <>
    Operator* CreateOp<CPU>(PredictionParam &param, long *size) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new PredictionOp<DType, CPU>(param);
            int shape = 0;
            for (Shape s : param.outShapes) {
                shape += s.Size();
            }
            *size = sizeof(DType) * shape;
        })
        return op;
    }

    template <>
    Operator* CreateOp<GPU>(PredictionParam &param, long *size) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new PredictionOp<DType, GPU>(param);
            int shape = 0;
            for (Shape s : param.outShapes) {
                shape += s.Size();
            }
            *size = sizeof(DType) * shape;
        })
        return op;
    }



    PredictionOpProp::PredictionOpProp() {
        param = new PredictionParam(kFloat);
    }

    PredictionOpProp::PredictionOpProp(const MatrixType &type) {
        param = new PredictionParam(type);
    }

    PredictionOpProp::~PredictionOpProp() {
        delete param;
    }

    void PredictionOpProp::InferShape(std::vector<Shape> &inShape, std::vector<Shape> &outShape) {

    }

    Operator *PredictionOpProp::CreateOperator(Context context, std::vector<Blob> &input, std::vector<Blob> &output,
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