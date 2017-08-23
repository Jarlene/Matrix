//
// Created by Jarlene on 2017/8/2.
//

#include "matrix/include/op/PoolingOp.h"

namespace matrix {

    template <class T, class Context>
    PoolingOp<T, Context>::PoolingOp(Parameter &param) {

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
    Operator* CreateOp<CPU>(PoolingParam &param, long *size) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new PoolingOp<DType, CPU>(param);
            int shape = 0;
            for (Shape s : param.outShapes) {
                shape += s.Size();
            }
            *size = sizeof(DType) * shape;
        })
        return op;
    }

    template <>
    Operator* CreateOp<GPU>(PoolingParam &param, long *size) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new PoolingOp<DType, GPU>(param);
            int shape = 0;
            for (Shape s : param.outShapes) {
                shape += s.Size();
            }
            *size = sizeof(DType) * shape;
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
        BIND_DISPATCH(CreateOp, *param, &memorySize);
    }
}