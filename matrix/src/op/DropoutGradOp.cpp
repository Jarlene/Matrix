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
        bool isTrain = this->context->phase == TRAIN;
        const T * data = Input<T>(DATA);
        T *out = Output<T>();
        if (isTrain) {
            T * mask = InputNonConst<T>(MASK);
            Random<T>(outputShape->Size(), mask, T(0), T(1.0));
            float rate = GetArgValue<float>("rate", 0.5f);
            for (int i = 0; i < outputShape->Size(); ++i) {
                if (mask[i] < rate) {
                    mask[i] = T(0);
                    out[i] = 0;
                } else {
                    mask[i] = T(1);
                    out[i] = data[i];
                }
            }
        } else {
            FallThrow();
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