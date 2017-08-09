//
// Created by Jarlene on 2017/8/2.
//

#include "matrix/include/op/PoolingOp.h"

namespace matrix {

    template <class T, class Context>
    PoolingOp<T, Context>::PoolingOp(PoolingParam &param) {

    }

    template <class T, class Context>
    bool PoolingOp<T, Context>::Run() {
        return Operator::Run();
    }

    template <class T, class Context>
    void PoolingOp<T, Context>::AsyncRun() {
        Operator::AsyncRun();
    }

    template <class T, class Context>
    bool PoolingOp<T, Context>::RunOnDevice() {
        return false;
    }

    template <class T, class Context>
    PoolingOp<T, Context>::~PoolingOp() {

    }



    template <>
    Operator* CreateOp<CPU>(PoolingParam &param) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new PoolingOp<DType, CPU>(param);
        })
        return op;
    }

    template <>
    Operator* CreateOp<GPU>(PoolingParam &param) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new PoolingOp<DType, GPU>(param);
        })
        return op;
    }

    PoolingParam::PoolingParam(MatrixType matrixType) : Parameter(matrixType) {

    }

    PoolingOpProp::PoolingOpProp() {
        param = new PoolingParam(kFloat);
    }

    PoolingOpProp::PoolingOpProp(const MatrixType &type) {
        param = new PoolingParam(type);
    }

    PoolingOpProp::~PoolingOpProp() {
        delete param;
    }

    void PoolingOpProp::InferShape(std::vector<Shape> &inShape, std::vector<Shape> &outShape) {

    }

    Operator *PoolingOpProp::CreateOperator(Context context, std::vector<Blob> &input, std::vector<Blob> &output,
                                            std::vector<Shape> &inShape, std::vector<Shape> &outShape,
                                            std::map<std::string, Any> &args) {
        InferShape(inShape, outShape);
        param->outputs = output;
        param->inputs = input;
        param->inputShapes = inShape;
        param->outShapes = outShape;
        param->args = args;
        BIND_DISPATCH(CreateOp, *param);
    }
}