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





    void FlattenOpProp::InferShape(std::vector<Shape *> &inShape, Shape *outShape) {
        outShape->reShape(ShapeN(inShape[0]->At(0), inShape[0]->StrideExclude(0)));
    }

    INIT_OPERATOR_PROPERTY_CREATE(FlattenOpProp, FlattenOp, false);

}

