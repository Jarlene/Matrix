//
// Created by Jarlene on 2017/8/2.
//

#include "matrix/include/op/GRUOp.h"

namespace matrix {

    template <class T, class Context>
    GRUOp<T, Context>::GRUOp(GRUParam &param) {

    }

    template <class T, class Context>
    bool GRUOp<T, Context>::Run() {
        return Operator::Run();
    }

    template <class T, class Context>
    void GRUOp<T, Context>::AsyncRun() {
        Operator::AsyncRun();
    }

    template <class T, class Context>
    bool GRUOp<T, Context>::RunOnDevice() {
        return false;
    }

    template <class T, class Context>
    GRUOp<T, Context>::~GRUOp() {

    }



    template <>
    Operator* CreateOp<CPU>(GRUParam &param) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new GRUOp<DType, CPU>(param);
        })
        return op;
    }

    template <>
    Operator* CreateOp<GPU>(GRUParam &param) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new GRUOp<DType, GPU>(param);
        })
        return op;
    }

    GRUParam::GRUParam(MatrixType matrixType) : Parameter(matrixType) {

    }

    GRUOpProp::GRUOpProp() {
        param = new GRUParam(kFloat);
    }

    GRUOpProp::GRUOpProp(const MatrixType &type) {
        param = new GRUParam(type);
    }

    GRUOpProp::~GRUOpProp() {
        delete param;
    }

    void GRUOpProp::InferShape(std::vector<Shape> &inShape, std::vector<Shape> &outShape) {

    }

    Operator *GRUOpProp::CreateOperator(Context context, std::vector<Blob> &input, std::vector<Blob> &output,
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