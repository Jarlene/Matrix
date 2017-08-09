//
// Created by Jarlene on 2017/8/2.
//

#include "matrix/include/op/LSTMOp.h"

namespace matrix {

    template <class T, class Context>
    LSTMOp<T, Context>::LSTMOp(LSTMParam &param) {

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
    Operator* CreateOp<CPU>(LSTMParam &param) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new LSTMOp<DType, CPU>(param);
        })
        return op;
    }

    template <>
    Operator* CreateOp<GPU>(LSTMParam &param) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new LSTMOp<DType, GPU>(param);
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

    void LSTMOpProp::InferShape(std::vector<Shape> &inShape, std::vector<Shape> &outShape) {

    }

    Operator *LSTMOpProp::CreateOperator(Context context, std::vector<Blob> &input, std::vector<Blob> &output,
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