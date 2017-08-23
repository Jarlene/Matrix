//
// Created by Jarlene on 2017/8/2.
//

#include "matrix/include/op/MulOp.h"

namespace matrix {


    template <class T, class Context>
    MulOp<T, Context>::MulOp(Parameter &param) {

    }

    template <class T, class Context>
    bool MulOp<T, Context>::Run() {
        return Operator::Run();
    }

    template <class T, class Context>
    void MulOp<T, Context>::AsyncRun() {
        Operator::AsyncRun();
    }

    template <class T, class Context>
    bool MulOp<T, Context>::RunOnDevice() {
        return false;
    }

    template <class T, class Context>
    MulOp<T, Context>::~MulOp() {

    }



    template <>
    Operator* CreateOp<CPU>(MulParam &param, long *size) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new MulOp<DType, CPU>(param);
            int shape = 0;
            for (Shape s : param.outShapes) {
                shape += s.Size();
            }
            *size = sizeof(DType) * shape;
        })
        return op;
    }

    template <>
    Operator* CreateOp<GPU>(MulParam &param, long *size) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new MulOp<DType, GPU>(param);
            int shape = 0;
            for (Shape s : param.outShapes) {
                shape += s.Size();
            }
            *size = sizeof(DType) * shape;
        })
        return op;
    }


    MulOpProp::MulOpProp() {
        param = new MulParam(kFloat);
    }

    MulOpProp::MulOpProp(const MatrixType &type) {
        param = new MulParam(type);
    }

    MulOpProp::~MulOpProp() {
        delete param;
    }

    void MulOpProp::InferShape(std::vector<Shape> &inShape, std::vector<Shape> &outShape) {
        assert(inShape.size() >= 2);
        assert(outShape.size() >= 1);
        ProduceMulOpShape(inShape, outShape[0]);
    }

    Operator *MulOpProp::CreateOperator(Context context, std::vector<Blob> &input, std::vector<Blob> &output,
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