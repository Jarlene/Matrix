//
// Created by Jarlene on 2017/11/5.
//

#include "matrix/include/op/FlattenOp.h"

namespace matrix {


    template <class T, class Context>
    FlattenOp<T, Context>::FlattenOp(Parameter &param) {
        INIT_PARAMS
    }

    template <class T, class Context>
    bool FlattenOp<T, Context>::Run() {
        FallThrow();
        return true;
    }

    template <class T, class Context>
    void FlattenOp<T, Context>::AsyncRun() {
        if (Context::mode == RunMode::kCpu) {
            Run();
        } else {
            if (!RunOnDevice()) {
                Run();
            }
        }
    }

    template <class T, class Context>
    bool FlattenOp<T, Context>::RunOnDevice() {
        return false;
    }

    template <class T, class Context>
    FlattenOp<T, Context>::~FlattenOp() {

    }




    FlattenOpProp::FlattenOpProp() {
        param = new Parameter(kFloat);
    }

    FlattenOpProp::FlattenOpProp(const MatrixType &type) {
        param = new Parameter(type);
    }

    FlattenOpProp::~FlattenOpProp() {
        delete param;
    }

    void FlattenOpProp::InferShape(std::vector<Shape *> &inShape, Shape *outShape) {
        outShape->reShape(ShapeN(inShape[0]->At(0), inShape[0]->StrideExclude(0)));
    }

    Operator *FlattenOpProp::CreateOperator(Context context, std::vector<void *> *input, void *output,
                                            std::vector<Shape *> *inShape, Shape *outShape,
                                            std::map<std::string, Any> &args) {
        param->args = &args;
        param->output = output;
        InferShape(*inShape, outShape);
        param->inputShapes = inShape;
        param->inputs = input;
        param->outShape = outShape;
        CREATE_OPERATOR(context, param, FlattenOp)
    }

}

