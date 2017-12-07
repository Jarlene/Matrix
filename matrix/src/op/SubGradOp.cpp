//
// Created by Jarlene on 2017/12/7.
//

#include "matrix/include/op/SubGradOp.h"




namespace matrix {
    template <class T, class xpu>
    SubGradOp<T, xpu>::SubGradOp(Parameter &param) {
        INIT_PARAMS
    }

    template <class T, class xpu>
    bool SubGradOp<T, xpu>::Run() {
        Tensor<T> in1(Input<T>(PRE_GRAD), *inputShapes->at(PRE_GRAD));
        Tensor<T> out(Output<T>(), *outputShape);
        Copy(in1, out);
        if (!HasArg("input_idx")) {
            Logger::Global()->Fatal("SubGradOp not support.");
        }
        int idx = GetArgValue<int>("input_idx");
        if (idx == 1) {
            Scale(out, T(-1));
        }
        return true;
    }

    template <class T, class xpu>
    void SubGradOp<T, xpu>::AsyncRun() {
        if (xpu::mode == RunMode::kCpu) {
            Run();
        } else {
            if (!RunOnDevice()) {
                Run();
            }
        }
    }

    template <class T, class xpu>
    bool SubGradOp<T, xpu>::RunOnDevice() {
        return false;
    }

    template <class T, class xpu>
    SubGradOp<T, xpu>::~SubGradOp() {

    }



    void SubGradOpProp::InferShape(std::vector<Shape*> &inShape, Shape* outShape) {
        assert(outShape != nullptr);
        outShape->reShape(*inShape.at(0));
    }


    INIT_OPERATOR_PROPERTY_CREATE(SubGradOpProp, SubGradOp, true);

}