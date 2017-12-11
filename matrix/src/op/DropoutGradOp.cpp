//
// Created by 郑珊 on 2017/12/2.
//

#include "matrix/include/op/DropoutGradOp.h"

namespace matrix {
    template <class T, class xpu>
    DropoutGradOp<T, xpu>::DropoutGradOp(Parameter &param) {
        INIT_PARAMS
    }

    template <class T, class xpu>
    bool DropoutGradOp<T, xpu>::Run() {
        const T * grad = Input<T>(PRE_GRAD);
        T *out = Output<T>();
        const T * mask = Input<T>(MASK);
        for (int i = 0; i < outputShape->Size(); ++i) {
            out[i] = grad[i] * mask[i];
        }
        return true;
    }

    template <class T, class xpu>
    void DropoutGradOp<T, xpu>::AsyncRun() {
        if (xpu::mode == RunMode::kCpu) {
            Run();
        } else {
            if (!RunOnDevice()) {
                Run();
            }
        }
    }

    template <class T, class xpu>
    bool DropoutGradOp<T, xpu>::RunOnDevice() {
        return false;
    }

    template <class T, class xpu>
    DropoutGradOp<T, xpu>::~DropoutGradOp() {

    }




    void DropoutGradOpProp::InferShape(std::vector<Shape*> &inShape, Shape* outShape) {
        if (param->context->phase == TRAIN) {
            outShape->reShape(*inShape[0]);
            return;
        }
        outShape->reShape(ShapeN(0));
    }

    INIT_OPERATOR_PROPERTY_CREATE(DropoutGradOpProp, DropoutGradOp, true);


}