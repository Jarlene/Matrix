//
// Created by Jarlene on 2017/8/2.
//

#include "matrix/include/op/DropoutOp.h"

namespace matrix {

    template <class T, class xpu>
    DropoutOp<T, xpu>::DropoutOp(Parameter &param) {
        INIT_PARAMS
    }

    template <class T, class xpu>
    bool DropoutOp<T, xpu>::Run() {
        bool isTrain = this->context->phase == TRAIN;
        const T * data = Input<T>(DATA);
        T *out = Output<T>();
        if (isTrain) {
            T * mask = Input<T>(MASK);
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
            CPUCopy(inputShapes->at(DATA)->Size(), Input<T>(DATA), 1, Output<T>(), 1);
        }
        return true;
    }

    template <class T, class xpu>
    void DropoutOp<T, xpu>::AsyncRun() {
        if (xpu::mode == RunMode::kCpu) {
            Run();
        } else {
            if (!RunOnDevice()) {
                Run();
            }
        }
    }

    template <class T, class xpu>
    bool DropoutOp<T, xpu>::RunOnDevice() {
        return false;
    }

    template <class T, class xpu>
    DropoutOp<T, xpu>::~DropoutOp() {

    }

    template <class T, class xpu>
    bool DropoutOp<T, xpu>::ShareNodes(std::function<void(std::initializer_list<Shape *> shapes)> func) {
        bool isTrain = this->context->phase == TRAIN;
        if (InputSize() == 1 && isTrain) {
            Shape mask;
            mask.reShape(*inputShapes->at(0));
            func({&mask});
            return true;
        }
        return false;
    }


    void DropoutOpProp::InferShape(std::vector<Shape*> &inShape, Shape* outShape) {
        if (param->context->phase == TRAIN) {
            outShape->reShape(*inShape[0]);
            return;
        }
        outShape->reShape(ShapeN(0));
    }

    INIT_OPERATOR_PROPERTY_CREATE(DropoutOpProp, DropoutOp, true);


}