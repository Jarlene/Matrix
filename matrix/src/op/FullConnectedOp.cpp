//
// Created by Jarlene on 2017/8/2.
//

#include "matrix/include/op/FullConnectedOp.h"

namespace matrix {

    template <class T, class xpu>
    FullConnectedOp<T, xpu>::FullConnectedOp(Parameter &param) {
        INIT_PARAMS
    }

    template <class T, class xpu>
    bool FullConnectedOp<T, xpu>::Run() {
        Tensor<T> data(Input<T>(DATA), *inputShapes->at(DATA));
        Tensor<T> weight(Input<T>(WEIGHT), *inputShapes->at(WEIGHT));
        Tensor<T> out(Output<T>(), *outputShape);
        MatrixMul<T>(data, false, weight, false, out);
         if (InputSize()  == 3) {
            Tensor<T> bias(Input<T>(BIAS), *inputShapes->at(BIAS));
            Add<T>(out, bias, out);
        }
        if (HasArg("activation_type")) {
            auto actType = GetArgValue<ActType>("activation_type");
            switch (actType) {
                case kSigmoid:
                    Sigmoid<T>(out, out);
                    break;
                case kTanh:
                    Tanh<T>(out, out);
                    break;
                case kRelu:
                    Relu<T>(out, out);
                    break;
                default:
                    Logger::Global()->Fatal("FullConnectedOp activation_type not support \n");
                    break;
            }
        }
        return true;
    }

    template <class T, class xpu>
    void FullConnectedOp<T, xpu>::AsyncRun() {
        if (xpu::mode == RunMode::kCpu) {
            Run();
        } else {
            if (!RunOnDevice()) {
                Run();
            }
        }
    }

    template <class T, class xpu>
    FullConnectedOp<T, xpu>::~FullConnectedOp() {

    }

    template <class T, class xpu>
    bool FullConnectedOp<T, xpu>::RunOnDevice() {
        return false;
    }

    template <class T, class xpu>
    bool FullConnectedOp<T, xpu>::VariableNode(std::function<void(std::initializer_list<Shape *> shapes)> func) {
        if (InputSize() == 1) {
            Shape weight;
            int rank = inputShapes->at(0)->Rank();
            for (int i = 0; i < rank - 2; ++i) {
                weight.Append(inputShapes->at(0)->At(i));
            }
            weight.Append(inputShapes->at(0)->At(rank - 1));
            if (!HasArg("hide_num")) {
                Logger::Global()->Fatal("FullConnectedOp can not find hide_num params");
            }
            weight.Append(GetArgValue<int>("hide_num"));
            if(HasArg("with_bias") && GetArgValue<bool>("with_bias")) {
                Shape bias;
                bias.Append(GetArgValue<int>("hide_num"));
                func({&weight, &bias});
            } else {
                func({&weight});
            }
            return true;
        }
        return false;
    };




    void FullConnectedOpProp::InferShape(std::vector<Shape*> &inShape, Shape *outShape) {
        if (inShape.size() == 1) {
            return;
        }
        assert(inShape.size() >= 2);
        assert(outShape != nullptr);
        ProduceMulOpShape(inShape, outShape);
    }

    INIT_OPERATOR_PROPERTY_CREATE(FullConnectedOpProp, FullConnectedOp, true);


}