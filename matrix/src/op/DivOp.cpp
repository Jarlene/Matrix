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




    DivOpProp::DivOpProp() {
        param = new Parameter(kFloat);
    }

    DivOpProp::DivOpProp(const MatrixType &type) {
        param = new Parameter(type);
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
        CREATE_OPERATOR(param, DivOp, {
            memorySize = sizeof(DType) * param->outShapes->Size();
        })
    }


}