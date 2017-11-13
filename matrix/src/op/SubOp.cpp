//
// Created by Jarlene on 2017/8/2.
//

#include "matrix/include/op/SubOp.h"

namespace matrix {
    template <class T, class Context>
    SubOp<T, Context>::SubOp(Parameter &param) {
        INIT_PARAMS
    }

    template <class T, class Context>
    bool SubOp<T, Context>::Run() {
        return Operator::Run();
    }

    template <class T, class Context>
    void SubOp<T, Context>::AsyncRun() {
        Operator::AsyncRun();
    }

    template <class T, class Context>
    bool SubOp<T, Context>::RunOnDevice() {
        return false;
    }

    template <class T, class Context>
    SubOp<T, Context>::~SubOp() {

    }



    SubOpProp::SubOpProp() {
        param = new Parameter(kFloat);
    }

    SubOpProp::SubOpProp(const MatrixType &type) {
        param = new Parameter(type);
    }

    SubOpProp::~SubOpProp() {
        delete param;
    }

    void SubOpProp::InferShape(std::vector<Shape*> &inShape, Shape* outShape) {

    }

    Operator *SubOpProp::CreateOperator(Context context, std::vector<void *> *input, void *output,
                                        std::vector<Shape *> *inShape, Shape *outShape,
                                        std::map<std::string, Any> &args) {
        param->args = &args;
        param->output = output;
        InferShape(*inShape, outShape);
        param->inputShapes = inShape;
        param->inputs = input;
        param->outShape = outShape;
        CREATE_OPERATOR(context, param, SubOp, {
            memorySize = sizeof(DType) * param->outShape->Size();
        })
    }

}