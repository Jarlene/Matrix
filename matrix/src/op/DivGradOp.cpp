//
// Created by Jarlene on 2017/12/7.
//

#include "matrix/include/op/DivGradOp.h"



namespace matrix {

    template <class T, class xpu>
    DivGradOp<T, xpu>::DivGradOp(Parameter &param) {
        INIT_PARAMS
    }

    template <class T, class xpu>
    bool DivGradOp<T, xpu>::Run() {
        Tensor<T> pre(Input<T>(PRE_GRAD), *inputShapes->at(PRE_GRAD));
        Tensor<T> in1(Input<T>(INPUT1), *inputShapes->at(INPUT1));
        Tensor<T> in2(Input<T>(INPUT2), *inputShapes->at(INPUT2));
        Tensor<T> out(Output<T>(), *outputShape);
        if (!HasArg("input_idx")) {
            Logger::Global()->Fatal("DivGradOp not support.");
        }
        int idx = GetArgValue<int>("input_idx");
        if (idx == 0) {

        } else if (idx == 1) {

        }
        return true;
    }

    template <class T, class xpu>
    void DivGradOp<T, xpu>::AsyncRun() {
        if (xpu::mode == RunMode::kCpu) {
            Run();
        } else {
            if (!RunOnDevice()) {
                Run();
            }
        }
    }

    template <class T, class xpu>
    DivGradOp<T, xpu>::~DivGradOp() {

    }

    template <class T, class xpu>
    bool DivGradOp<T, xpu>::RunOnDevice() {
        return false;
    }




    void DivGradOpProp::InferShape(std::vector<Shape*> &inShape, Shape *outShape) {
        assert(inShape.size() >= 2);
        assert(outShape != nullptr);
        ProduceMulOpShape(inShape, outShape);
    }


    INIT_OPERATOR_PROPERTY_CREATE(DivGradOpProp, DivGradOp, true);

}