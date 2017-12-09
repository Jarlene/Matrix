//
// Created by Jarlene on 2017/11/9.
//

#include "matrix/include/op/FlattenGradOp.h"

namespace matrix {
    template <class T, class xpu>
    FlattenGradOp<T, xpu>::FlattenGradOp(Parameter &param) {
        INIT_PARAMS
    }

    template <class T, class xpu>
    bool FlattenGradOp<T, xpu>::Run() {
        CPUCopy(inputShapes->at(PRE_GRAD)->Size(), Input<T>(PRE_GRAD), 1, Output<T>(), 1);
        return true;
    }

    template <class T, class xpu>
    void FlattenGradOp<T, xpu>::AsyncRun() {
        if (xpu::mode == RunMode::kCpu) {
            Run();
        } else {
            if (!RunOnDevice()) {
                Run();
            }
        }
    }

    template <class T, class xpu>
    bool FlattenGradOp<T, xpu>::RunOnDevice() {
        return false;
    }

    template <class T, class xpu>
    FlattenGradOp<T, xpu>::~FlattenGradOp() {

    }




    void FlattenGradOpProp::InferShape(std::vector<Shape *> &inShape, Shape *outShape) {
        outShape->reShape(*inShape[2]);
    }


    INIT_OPERATOR_PROPERTY_CREATE(FlattenGradOpProp, FlattenGradOp, true);

}