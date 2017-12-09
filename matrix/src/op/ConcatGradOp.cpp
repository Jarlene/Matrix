//
// Created by Jarlene on 2017/11/21.
//

#include "matrix/include/op/ConcatGradOp.h"


namespace matrix {
    template <class T, class xpu>
    ConcatGradOp<T, xpu>::ConcatGradOp(Parameter &param) {
        INIT_PARAMS
    }

    template <class T, class xpu>
    bool ConcatGradOp<T, xpu>::Run() {
        int idx = GetArgValue<int>("input_idx", -1);
        if(idx == -1) {
            Logger::Global()->Fatal("ConcatGradOp need input idx");
        }
        Tensor<T> out(Output<T>(), *outputShape);
        Tensor<T> pre_grad(Input<T>(idx+2), *inputShapes->at(idx+2));
        CPUCopy(pre_grad.Size(), pre_grad.Data(), 1, out.MutableData(), 1);
        return true;
    }

    template <class T, class xpu>
    void ConcatGradOp<T, xpu>::AsyncRun() {
        if (xpu::mode == RunMode::kCpu) {
            Run();
        } else {
            if (!RunOnDevice()) {
                Run();
            }
        }
    }

    template <class T, class xpu>
    bool ConcatGradOp<T, xpu>::RunOnDevice() {
        return false;
    }

    template <class T, class xpu>
    ConcatGradOp<T, xpu>::~ConcatGradOp() {

    }



    void ConcatGradOpProp::InferShape(std::vector<Shape*> &inShape, Shape* outShape) {
        assert(outShape != nullptr);
        if (!param->args->count("input_idx")) {
            Logger::Global()->Fatal("ConcatGradOpProp need input idx");
        }
        int idx = get<int>(param->args->at("input_idx"));
        outShape->reShape(*inShape.at(idx));

    }

    INIT_OPERATOR_PROPERTY_CREATE(ConcatGradOpProp, ConcatGradOp, true);

}