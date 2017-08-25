//
// Created by Jarlene on 2017/8/2.
//

#include "matrix/include/op/LSTMOp.h"

namespace matrix {

    template <class T, class Context>
    LSTMOp<T, Context>::LSTMOp(Parameter &param) {
        INIT_PARAMS
    }

    template <class T, class Context>
    bool LSTMOp<T, Context>::Run() {
        return Operator::Run();
    }

    template <class T, class Context>
    void LSTMOp<T, Context>::AsyncRun() {
        Operator::AsyncRun();
    }

    template <class T, class Context>
    bool LSTMOp<T, Context>::RunOnDevice() {
        return false;
    }

    template <class T, class Context>
    LSTMOp<T, Context>::~LSTMOp() {

    }



    template <>
    Operator* CreateOp<CPU>(LSTMParam &param, long *size) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new LSTMOp<DType, CPU>(param);
            *size = sizeof(DType) * param.outShapes->Size();
        })
        return op;
    }

    template <>
    Operator* CreateOp<GPU>(LSTMParam &param, long *size) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new LSTMOp<DType, GPU>(param);
            *size = sizeof(DType) * param.outShapes->Size();
        })
        return op;
    }

    LSTMParam::LSTMParam(MatrixType matrixType) : Parameter(matrixType) {

    }

    LSTMOpProp::LSTMOpProp() {
        param = new LSTMParam(kFloat);
    }

    LSTMOpProp::LSTMOpProp(const MatrixType &type) {
        param = new LSTMParam(type);
    }

    LSTMOpProp::~LSTMOpProp() {
        delete param;
    }

    void LSTMOpProp::InferShape(std::vector<Shape*> &inShape, Shape *outShape) {

    }

    Operator *LSTMOpProp::CreateOperator(Context context, std::vector<Blob*> &input, Blob* output,
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

    void LSTMOpProp::SwitchType(const MatrixType &type) {
        this->param->type = type;
    }
}