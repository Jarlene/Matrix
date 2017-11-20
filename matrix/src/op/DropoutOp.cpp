//
// Created by Jarlene on 2017/8/2.
//

#include "matrix/include/op/DropoutOp.h"

namespace matrix {

    template <class T, class Context>
    DropoutOp<T, Context>::DropoutOp(Parameter &param) {
        INIT_PARAMS
    }

    template <class T, class Context>
    bool DropoutOp<T, Context>::Run() {
        bool isTrain = GetArgValue<bool>("isTrain", false);
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

    template <class T, class Context>
    void DropoutOp<T, Context>::AsyncRun() {
        if (Context::mode == RunMode::kCpu) {
            Run();
        } else {
            if (!RunOnDevice()) {
                Run();
            }
        }
    }

    template <class T, class Context>
    bool DropoutOp<T, Context>::RunOnDevice() {
        return false;
    }

    template <class T, class Context>
    DropoutOp<T, Context>::~DropoutOp() {

    }

    template <class T, class Context>
    bool DropoutOp<T, Context>::ShareNodes(std::function<void(std::initializer_list<Shape *> shapes)> func) {
        bool isTrain = GetArgValue<bool>("isTrain", false);
        if (InputSize() == 1 && isTrain) {
            Shape mask;
            mask.reShape(*inputShapes->at(0));
            func({&mask});
            return true;
        }
        return false;
    }


    void DropoutOpProp::InferShape(std::vector<Shape*> &inShape, Shape* outShape) {
        if (param->args->count("isTrain")) {
            if (get<bool>(param->args->at("isTrain"))) {
                outShape->reShape(*inShape[0]);
                return;
            }
        }
        outShape->reShape(ShapeN(0));
    }

    INIT_OPERATOR_PROPERTY_CREATE(DropoutOpProp, DropoutOp, true);


}