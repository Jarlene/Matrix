//
// Created by Jarlene on 2017/11/9.
//

#include "matrix/include/op/FlattenGradOp.h"

namespace matrix {
    template <class T, class Context>
    FlattenGradOp<T, Context>::FlattenGradOp(Parameter &param) {
        INIT_PARAMS
    }

    template <class T, class Context>
    bool FlattenGradOp<T, Context>::Run() {
        FallThrow();
        return true;
    }

    template <class T, class Context>
    void FlattenGradOp<T, Context>::AsyncRun() {
        if (Context::mode == RunMode::kCpu) {
            Run();
        } else {
            if (!RunOnDevice()) {
                Run();
            }
        }
    }

    template <class T, class Context>
    bool FlattenGradOp<T, Context>::RunOnDevice() {
        return false;
    }

    template <class T, class Context>
    FlattenGradOp<T, Context>::~FlattenGradOp() {

    }



    FlattenGradOpProp::FlattenGradOpProp() {
        param = new Parameter(kFloat);
    }

    FlattenGradOpProp::FlattenGradOpProp(const MatrixType &type) {
        param = new Parameter(type);
    }

    FlattenGradOpProp::~FlattenGradOpProp() {
        delete param;
    }

    void FlattenGradOpProp::InferShape(std::vector<Shape *> &inShape, Shape *outShape) {
        outShape->reShape(ShapeN(inShape[0]->At(0), inShape[0]->StrideExclude(0)));
    }

    Operator *FlattenGradOpProp::CreateOperator(Context context, std::vector<void *> &input, void *output,
                                                std::vector<Shape *> &inShape, Shape *outShape,
                                                std::map<std::string, Any> &args) {
        param->args = &args;
        param->inputs = input;
        param->output = output;
        InferShape(inShape, outShape);
        param->inputShapes = inShape;
        param->outShape = outShape;
        CREATE_OPERATOR(param, FlattenGradOp)

    }


}