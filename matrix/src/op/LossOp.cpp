//
// Created by Jarlene on 2017/8/2.
//

#include "matrix/include/op/LossOp.h"


namespace matrix {

    template <class T, class Context>
    LossOp<T, Context>::LossOp(LossParam &param) {

    }

    template <class T, class Context>
    bool LossOp<T, Context>::Run() {
        return Operator::Run();
    }

    template <class T, class Context>
    void LossOp<T, Context>::AsyncRun() {
        Operator::AsyncRun();
    }

    template <class T, class Context>
    LossOp<T, Context>::~LossOp() {

    }

    template <class T, class Context>
    bool LossOp<T, Context>::RunOnDevice() {
        return false;
    }




    template <>
    Operator* CreateOp<CPU>(LossParam &param) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new LossOp<DType, CPU>(param);
        })
        return op;
    }

    template <>
    Operator* CreateOp<GPU>(LossParam &param) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new LossOp<DType, GPU>(param);
        })
        return op;
    }

    LossParam::LossParam(MatrixType matrixType) : Parameter(matrixType) {

    }

    LossOpProp::LossOpProp() {
        param = new LossParam(kFloat);
    }

    LossOpProp::LossOpProp(const MatrixType &type) {
        param = new LossParam(type);
    }

    LossOpProp::~LossOpProp() {
        delete param;
    }

    void LossOpProp::InferShape(std::vector<Shape> &inShape, std::vector<Shape> &outShape) {

    }

    Operator *LossOpProp::CreateOperator(Context context, std::vector<Blob> &input, std::vector<Blob> &output,
                                         std::vector<Shape> &inShape, std::vector<Shape> &outShape) {
        InferShape(inShape, outShape);
        param->inputs = input;
        param->outputs = output;
        param->inputShapes = inShape;
        param->outShapes = outShape;
        BIND_DISPATCH(CreateOp, *param);
    }
}