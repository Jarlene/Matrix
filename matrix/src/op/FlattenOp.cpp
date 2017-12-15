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
        CPUCopy(inputShapes->at(INPUT)->Size(), Input<T>(INPUT), 1, Output<T>(), 1);
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
        if (param->args->count("shape")) {
            Shape target = get<Shape>(param->args->at("shape"));
            outShape->reShape(target);
            return;
        }
        outShape->reShape(ShapeN(inShape[0]->At(0), inShape[0]->StrideExclude(0)));
    }

    INIT_OPERATOR_PROPERTY_CREATE(FlattenOpProp, FlattenOp, true);

}

