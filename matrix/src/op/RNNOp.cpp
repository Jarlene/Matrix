//
// Created by Jarlene on 2017/11/9.
//

#include "matrix/include/op/RNNOp.h"

namespace matrix {


    template <class T, class Context>
    RNNOp<T, Context>::RNNOp(Parameter &param) {
        INIT_PARAMS
    }

    template <class T, class Context>
    bool RNNOp<T, Context>::Run() {
        return Operator::Run();
    }

    template <class T, class Context>
    void RNNOp<T, Context>::AsyncRun() {
        Operator::AsyncRun();
    }

    template <class T, class Context>
    bool RNNOp<T, Context>::RunOnDevice() {
        return false;
    }

    template <class T, class Context>
    RNNOp<T, Context>::~RNNOp() {

    }



    template <>
    Operator* CreateOp<CPU>(Parameter &param, long *size) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new RNNOp<DType, CPU>(param);
            *size = sizeof(DType) * param.outShapes->Size();
        })
        return op;
    }

    template <>
    Operator* CreateOp<GPU>(Parameter &param, long *size) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new RNNOp<DType, GPU>(param);
            *size = sizeof(DType) * param.outShapes->Size();
        })
        return op;
    }



    RNNOpProp::RNNOpProp() {
        param = new Parameter(kFloat);
    }

    RNNOpProp::RNNOpProp(const MatrixType &type) {
        param = new Parameter(type);
    }

    RNNOpProp::~RNNOpProp() {
        delete param;
    }

    void RNNOpProp::InferShape(std::vector<Shape*> &inShape, Shape *outShape) {

    }

    Operator *RNNOpProp::CreateOperator(Context context, std::vector<Blob*> &input, Blob* output,
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


}
