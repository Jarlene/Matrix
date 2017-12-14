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
        ++num_of_pass;
        Tensor<T> variable(Input<T>(VARIABLE), *inputShapes->at(VARIABLE));
        Tensor<T> grad_variable(Input<T>(GRAD_VARIABLE), *inputShapes->at(GRAD_VARIABLE));
        auto type = GetArgValue<ApplyGradMode>("type", kSGD);
        float decay = GetArgValue<float>("decay", 0.01f);
        float learning_rate = GetArgValue<float>("learning_rate", 0.01f);
        learning_rate *= 1.0f/(1.0f + decay * num_of_pass) ;
        switch (type) {
            case kSGD:
            {
                T learn = T(-1*learning_rate);
                ApplyNode<T>(variable, grad_variable, learn);
            }
                break;
            case kMomentum:
            {
                Tensor<T> momentum(Input<T>(MOMENTUM), *inputShapes->at(MOMENTUM));
                float mon = GetArgValue<float>("momentum", 0.9f);
                T learn = T(-1.0 * learning_rate);
                Scale(grad_variable, learn);
                applyMomentum<T>(momentum, grad_variable, T(1), T(mon));
                ApplyNode<T>(variable, momentum, T(1));
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

    template<class T, class xpu>
    bool UpdateOp<T, xpu>::ShareNodes(std::function<void(std::initializer_list<Shape *> shapes)> func) {
        auto type = GetArgValue<ApplyGradMode>("type", kSGD);
        bool result = false;
        switch (type) {
            case kSGD:
                break;
            case kMomentum:
                if (InputSize() == 2) {
                    Shape momentum;
                    momentum.reShape(*inputShapes->at(GRAD_VARIABLE));
                    func({&momentum});
                    result = true;
                }
                break;
            case kNesterov:
                result = true;
                break;
            case kAdagrad:
                result = true;
                break;
            case kAdadelta:
                result = true;
                break;
            case kRMSprop:
                result = true;
                break;
            case kAdam:
                result = true;
                break;
            case kAdamax:
                result = true;
                break;
            case kNadam:
                result = true;
                break;
        }
        return result;
    }


    void UpdateOpProp::InferShape(std::vector<Shape*> &inShape, Shape* outShape) {
    }

    INIT_OPERATOR_PROPERTY_CREATE(UpdateOpProp, UpdateOp, false);

}