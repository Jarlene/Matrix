//
// Created by Jarlene on 2017/8/9.
//

#include "matrix/include/op/PlaceHolderOp.h"

namespace matrix {

    template <class T, class xpu>
    PlaceHolderOp<T, xpu>::PlaceHolderOp(PlaceHolderParam &param) {

    }


    template <class T, class xpu>
    bool PlaceHolderOp<T, xpu>::Run() {
        return true;
    }

    template <class T, class xpu>
    void PlaceHolderOp<T, xpu>::AsyncRun() {

    }


    template <class T, class xpu>
    bool PlaceHolderOp<T, xpu>::RunOnDevice() {
        return false;
    }


    template <class T, class xpu>
    PlaceHolderOp<T, xpu>::~PlaceHolderOp() {

    }

    template <>
    Operator* CreateOp<CPU>(PlaceHolderParam &param) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new PlaceHolderOp<DType, CPU>(param);
        })
        return op;
    }

    template <>
    Operator* CreateOp<GPU>(PlaceHolderParam &param) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new PlaceHolderOp<DType, GPU>(param);
        })
        return op;
    }



    PlaceHolderOpProp::PlaceHolderOpProp() {
        param = new PlaceHolderParam(kFloat);
    }

    PlaceHolderOpProp::PlaceHolderOpProp(const MatrixType type) {
        param = new PlaceHolderParam(type);
    }

    PlaceHolderOpProp::~PlaceHolderOpProp() {
        delete param;
    }

    void PlaceHolderOpProp::InferShape(std::vector<Shape> &inShape, std::vector<Shape> &outShape) {

    }

    Operator *PlaceHolderOpProp::CreateOperator(Context context, std::vector<Blob> &input, std::vector<Blob> &output,
                                                std::vector<Shape> &inShape, std::vector<Shape> &outShape) {
        InferShape(inShape, outShape);
        param->outputs = output;
        param->inputs = input;
        param->inputShapes = inShape;
        param->outShapes = outShape;
        BIND_DISPATCH(CreateOp, *param);
    }
}