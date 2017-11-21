//
// Created by  Jarlene on 2017/8/9.
//

#include <matrix/include/optimizer/BaseOptimizer.h>
#include "matrix/include/op/UpdateOp.h"

namespace matrix {

    template <class T, class xpu>
    UpdateOp<T, xpu>::UpdateOp(Parameter &param) {
        INIT_PARAMS
    }


    template <class T, class xpu>
    bool UpdateOp<T, xpu>::Run() {
        if (InputSize() < 2) {
            Logger::Global()->Fatal("input size less then 2");
        }
        Tensor<T> variable(Input<T>(VARIABLE), *inputShapes->at(VARIABLE));
        Tensor<T> grad_variable(Input<T>(GRAD_VARIABLE), *inputShapes->at(GRAD_VARIABLE));
        auto type = GetArgValue<ApplyGradMode>("type", kSGD);
        float learning_rate = GetArgValue<float>("learning_rate", 0.001f);
        switch (type) {
            case kSGD:
            {
                T learn = T(-1*learning_rate);
                ApplyNode<T>(variable, grad_variable, learn);
            }
                break;
            case kMomentum:
            {
                float mon = GetArgValue<float>("momentum", 0.9f);
                T learn = T(-1*learning_rate);
                Scale(variable, T(1 + mon));
                ApplyNode<T>(variable, grad_variable, learn);
            }
                break;
            case kNesterov:
            {

            }
                break;
            case kAdagrad:
            {

            }
                break;
            case kAdadelta:
            {

            }
                break;
            case kRMSprop:
            {

            }
                break;
            case kAdam:
            {

            }
                break;
            case kAdamax:
            {

            }
                break;
            case kNadam:
            {

            }
                break;
            default:
            Logger::Global()->Fatal("UpdateOp can not support");
                break;
        }
        return true;

    }

    template <class T, class xpu>
    void UpdateOp<T, xpu>::AsyncRun() {
        if (xpu::mode == RunMode::kCpu) {
            Run();
        } else if (xpu::mode == RunMode::kGpu) {
            if (!RunOnDevice()) {
                Run();
            }
        }
    }

    template <class T, class xpu>
    UpdateOp<T, xpu>::~UpdateOp() {

    }

    template <class T, class xpu>
    bool UpdateOp<T, xpu>::RunOnDevice() {
        return false;
    }


    void UpdateOpProp::InferShape(std::vector<Shape*> &inShape, Shape* outShape) {
    }

    INIT_OPERATOR_PROPERTY_CREATE(UpdateOpProp, UpdateOp, false);

}