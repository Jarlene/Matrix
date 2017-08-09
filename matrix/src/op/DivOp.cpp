//
// Created by Jarlene on 2017/8/2.
//

#include "matrix/include/op/DivOp.h"

namespace matrix {

    template <class T, class Context>
    DivOp<T, Context>::DivOp(DivParam &param) {

    }

    template <class T, class Context>
    bool DivOp<T, Context>::Run() {
        return Operator::Run();
    }

    template <class T, class Context>
    void DivOp<T, Context>::AsyncRun() {
        Operator::AsyncRun();
    }

    template <class T, class Context>
    DivOp<T, Context>::~DivOp() {

    }

    template <class T, class Context>
    bool DivOp<T, Context>::RunOnDevice() {
        return false;
    }




    template <>
    Operator* CreateOp<CPU>(DivParam &param) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new DivOp<DType, CPU>(param);
        })
        return op;
    }

    template <>
    Operator* CreateOp<GPU>(DivParam &param) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new DivOp<DType, GPU>(param);
        })
        return op;
    }

    DivParam::DivParam(MatrixType matrixType) : Parameter(matrixType) {

    }

    DivOpProp::DivOpProp() {
        param = new DivParam(kFloat);
    }

    DivOpProp::DivOpProp(const MatrixType &type) {
        param = new DivParam(type);
    }

    DivOpProp::~DivOpProp() {
        delete param;
    }

    void DivOpProp::InferShape(std::vector<Shape> &inShape, std::vector<Shape> &outShape) {

    }

    Operator *DivOpProp::CreateOperator(Context context, std::vector<Blob> &input, std::vector<Blob> &output,
                                        std::vector<Shape> &inShape, std::vector<Shape> &outShape,
                                        std::map<std::string, Any> &args) {
        InferShape(inShape, outShape);
        param->inputs = input;
        param->outputs = output;
        param->inputShapes = inShape;
        param->outShapes = outShape;
        param->args = args;
        BIND_DISPATCH(CreateOp, *param);
    }
}