//
// Created by Jarlene on 2017/8/2.
//

#include "matrix/include/op/MulOp.h"

namespace matrix {


    template <class T, class Context>
    MulOp<T, Context>::MulOp(Parameter &param) {
        INIT_PARAMS
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
            *size = sizeof(DType) * param.outShapes->Size();
        })
        return op;
    }

    template <>
    Operator* CreateOp<GPU>(MulParam &param, long *size) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new MulOp<DType, GPU>(param);
            *size = sizeof(DType) * param.outShapes->Size();
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

    void MulOpProp::InferShape(std::vector<Shape*> &inShape, Shape *outShape) {
        assert(inShape.size() >= 2);
        assert(outShape != nullptr);
        ProduceMulOpShape(inShape, outShape);
    }

    Operator *MulOpProp::CreateOperator(Context context, std::vector<Blob*> &input, Blob* output,
                                        std::vector<Shape*> &inShape, Shape *outShape,
                                        std::map<std::string, Any> &args) {
        param->args = &args;
        param->inputs = input;
        param->outputs = output;
        InferShape(inShape, outShape);
        param->inputShapes = inShape;
        param->outShapes = outShape;
        BIND_DISPATCH(CreateOp, *param, &memorySize);
    }

    void MulOpProp::SwitchType(const MatrixType &type) {
        this->param->type = type;
    }
}