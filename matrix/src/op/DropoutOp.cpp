//
// Created by Jarlene on 2017/8/2.
//

#include "matrix/include/op/DropoutOp.h"

namespace matrix {

    template <class T, class Context>
    DropoutOp<T, Context>::DropoutOp(DropoutParam &param) {

    }

    template <class T, class Context>
    bool DropoutOp<T, Context>::Run() {
        return Operator::Run();
    }

    template <class T, class Context>
    void DropoutOp<T, Context>::AsyncRun() {
        Operator::AsyncRun();
    }

    template <class T, class Context>
    bool DropoutOp<T, Context>::RunOnDevice() {
        return false;
    }

    template <class T, class Context>
    DropoutOp<T, Context>::~DropoutOp() {

    }



    template <>
    Operator* CreateOp<CPU>(DropoutParam &param) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new DropoutOp<DType, CPU>(param);
        })
        return op;
    }

    template <>
    Operator* CreateOp<GPU>(DropoutParam &param) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new DropoutOp<DType, GPU>(param);
        })
        return op;
    }

    DropoutParam::DropoutParam(MatrixType matrixType) : Parameter(matrixType) {

    }

    DropoutOpProp::DropoutOpProp() {
        param = new DropoutParam(kFloat);
    }

    DropoutOpProp::DropoutOpProp(const MatrixType &type) {
        param = new DropoutParam(type);
    }

    DropoutOpProp::~DropoutOpProp() {
        delete param;
    }

    void DropoutOpProp::InferShape(std::vector<Shape> &inShape, std::vector<Shape> &outShape) {

    }

    Operator *DropoutOpProp::CreateOperator(Context context, std::vector<Blob> &input, std::vector<Blob> &output,
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