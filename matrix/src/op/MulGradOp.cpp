//
// Created by Jarlene on 2017/11/15.
//

#include "matrix/include/op/MulGradOp.h"
namespace matrix {


    template <class T, class xpu>
    MulGradOp<T, xpu>::MulGradOp(Parameter &param) {
        INIT_PARAMS
    }

    template <class T, class xpu>
    bool MulGradOp<T, xpu>::Run() {
        if (!HasArg("input_idx")) {
            Logger::Global()->Fatal("MulGradOp not support.");
        }
        int idx = GetArgValue<int>("input_idx");

        Tensor<T> pre_grad(Input<T>(PRE_GRAD), *inputShapes->at(PRE_GRAD));
        Tensor<T> in1(Input<T>(INPUT1), *inputShapes->at(INPUT1));
        Tensor<T> in2(Input<T>(INPUT2), *inputShapes->at(INPUT2));
        Tensor<T> out(Output<T>(), *outputShape);
        if (idx == 0) {
            MatrixMul(pre_grad, false, in2, true, out);
        } else if (idx == 1) {
            MatrixMul(in2, true, pre_grad, false, out);
        }

        return true;
    }

    template <class T, class xpu>
    void MulGradOp<T, xpu>::AsyncRun() {
        if (xpu::mode == RunMode::kCpu) {
            Run();
        } else {
            if (!RunOnDevice()) {
                Run();
            }
        }
    }

    template <class T, class xpu>
    bool MulGradOp<T, xpu>::RunOnDevice() {
        return false;
    }

    template <class T, class xpu>
    MulGradOp<T, xpu>::~MulGradOp() {

    }



    void MulGradOpProp::InferShape(std::vector<Shape*> &inShape, Shape *outShape) {
        if (param->args->count("input_idx")) {
            int idx = get<int>(param->args->at("input_idx"));
            outShape->reShape(*inShape.at(idx + 2));
        }
    }

    INIT_OPERATOR_PROPERTY_CREATE(MulGradOpProp, MulGradOp, true);
}