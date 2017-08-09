//
// Created by Jarlene on 2017/8/2.
//

#include "matrix/include/op/FullConnectedOp.h"

namespace matrix {

    template <class T, class Context>
    FullConnectedOp<T, Context>::FullConnectedOp(FullConnectedParam &param) {

    }

    template <class T, class Context>
    bool FullConnectedOp<T, Context>::Run() {
        return Operator::Run();
    }

    template <class T, class Context>
    void FullConnectedOp<T, Context>::AsyncRun() {
        Operator::AsyncRun();
    }

    template <class T, class Context>
    FullConnectedOp<T, Context>::~FullConnectedOp() {

    }

    template <class T, class Context>
    bool FullConnectedOp<T, Context>::RunOnDevice() {
        return false;
    }




    template <>
    Operator* CreateOp<CPU>(FullConnectedParam &param, long *size) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new FullConnectedOp<DType, CPU>(param);
            int shape = 0;
            for (Shape s : param.outShapes) {
                shape += s.Size();
            }
            *size = sizeof(DType) * shape;
        })
        return op;
    }

    template <>
    Operator* CreateOp<GPU>(FullConnectedParam &param, long *size) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new FullConnectedOp<DType, GPU>(param);
            int shape = 0;
            for (Shape s : param.outShapes) {
                shape += s.Size();
            }
            *size = sizeof(DType) * shape;
        })
        return op;
    }

    FullConnectedParam::FullConnectedParam(MatrixType matrixType) : Parameter(matrixType) {

    }

    FullConnectedOpProp::FullConnectedOpProp() {
        param = new FullConnectedParam(kFloat);
    }

    FullConnectedOpProp::FullConnectedOpProp(const MatrixType &type) {
        param = new FullConnectedParam(type);
    }

    FullConnectedOpProp::~FullConnectedOpProp() {
        delete param;
    }

    void FullConnectedOpProp::InferShape(std::vector<Shape> &inShape, std::vector<Shape> &outShape) {

    }

    Operator *FullConnectedOpProp::CreateOperator(Context context, std::vector<Blob> &input, std::vector<Blob> &output,
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