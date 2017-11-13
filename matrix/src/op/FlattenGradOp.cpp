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




    void FlattenGradOpProp::InferShape(std::vector<Shape *> &inShape, Shape *outShape) {
        outShape->reShape(ShapeN(inShape[0]->At(0), inShape[0]->StrideExclude(0)));
    }


    INIT_OPERATOR_PROPERTY_CREATE(FlattenGradOpProp, FlattenGradOp, false);

}