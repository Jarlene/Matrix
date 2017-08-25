//
// Created by Jarlene on 2017/8/2.
//

#include "matrix/include/op/DivOp.h"

namespace matrix {

    template <class T, class Context>
    DivOp<T, Context>::DivOp(Parameter &param) {
        INIT_PARAMS
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
    Operator* CreateOp<CPU>(DivParam &param, long* size) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new DivOp<DType, CPU>(param);
            *size = sizeof(DType) * param.outShapes->Size();
        })
        return op;
    }

    template <>
    Operator* CreateOp<GPU>(DivParam &param, long *size) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new DivOp<DType, GPU>(param);
            *size = sizeof(DType) * param.outShapes->Size();
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

    void DivOpProp::InferShape(std::vector<Shape*> &inShape, Shape *outShape) {
        outShape->reShape(*inShape.at(0));
    }

    Operator *DivOpProp::CreateOperator(Context context, std::vector<Blob*> &input, Blob* output,
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

    void DivOpProp::SwitchType(const MatrixType &type) {
        this->param->type = type;
    }
}