//
// Created by Jarlene on 2017/7/31.
//

#include "matrix/include/op/AccuracyOp.h"


namespace matrix {


    template <class T, class Context>
    AccuracyOp<T, Context>::AccuracyOp(AccuracyParam &param) {

    }

    template <class T, class Context>
    bool AccuracyOp<T, Context>::RunOnDevice() {
        return false;
    }

    template <class T, class Context>
    bool AccuracyOp<T, Context>::Run() {
        return true;
    }

    template <class T, class Context>
    void AccuracyOp<T, Context>::AsyncRun() {
    }

    template <class T, class Context>
    AccuracyOp<T, Context>::~AccuracyOp() {

    }




    template <>
    Operator* CreateOp<CPU>(AccuracyParam &param) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new AccuracyOp<DType, CPU>(param);
        })
        return op;
    }

    template <>
    Operator* CreateOp<GPU>(AccuracyParam &param) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new AccuracyOp<DType, GPU>(param);
        })
        return op;
    }

    AccuracyOpProp::AccuracyOpProp()  {
        param = new AccuracyParam(MatrixType::kFloat);

    }

    AccuracyOpProp::AccuracyOpProp(const MatrixType type) {
        param = new AccuracyParam(type);
    }

    AccuracyOpProp::~AccuracyOpProp() {
        delete param;
    }

    void AccuracyOpProp::InferShape(std::vector<Shape> &inShape, std::vector<Shape> &outShape) {

    }

    Operator *AccuracyOpProp::CreateOperator(Context context, std::vector<Blob> &input, std::vector<Blob> &output,
                                             std::vector<Shape> &inShape, std::vector<Shape> &outShape) {
        InferShape(inShape, outShape);
        param->inputs = input;
        param->outputs = output;
        param->inputShapes = inShape;
        param->outShapes = outShape;
        BIND_DISPATCH(CreateOp, *param);
    }
}