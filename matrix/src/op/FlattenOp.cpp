//
// Created by Jarlene on 2017/11/5.
//

#include "matrix/include/op/FlattenOp.h"

namespace matrix {


    template <class T, class xpu>
    FlattenOp<T, xpu>::FlattenOp(Parameter &param) {
        INIT_PARAMS
    }

    template <class T, class xpu>
    bool FlattenOp<T, xpu>::Run() {
        FallThrow();
        return true;
    }

    template <class T, class xpu>
    void FlattenOp<T, xpu>::AsyncRun() {
        if (xpu::mode == RunMode::kCpu) {
            Run();
        } else {
            if (!RunOnDevice()) {
                Run();
            }
        }
    }

    template <class T, class xpu>
    bool FlattenOp<T, xpu>::RunOnDevice() {
        return false;
    }

    template <class T, class xpu>
    FlattenOp<T, xpu>::~FlattenOp() {

    }





    void FlattenOpProp::InferShape(std::vector<Shape *> &inShape, Shape *outShape) {
        outShape->reShape(ShapeN(inShape[0]->At(0), inShape[0]->StrideExclude(0)));
    }

    INIT_OPERATOR_PROPERTY_CREATE(FlattenOpProp, FlattenOp, true);

}

