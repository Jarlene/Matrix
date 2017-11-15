//
// Created by Jarlene on 2017/11/12.
//

#include "matrix/include/op/OutputGradOp.h"

namespace matrix {


    template <class T, class Context>
    OutputGradOp<T, Context>::OutputGradOp(Parameter &param) {
        INIT_PARAMS
    }

    template <class T, class Context>
    bool OutputGradOp<T, Context>::Run() {
        auto mode = GetArgValue<OutputMode>("type", kSoftmax);
        Tensor<T> pre_grad(Input<T>(PRE_GRAD), *inputShapes->at(PRE_GRAD));
        Tensor<T> selfOut(Input<T>(OUT), *inputShapes->at(OUT));
        Tensor<T> input(Input<T>(INPUT), *inputShapes->at(INPUT));
        Tensor<T> out(Output<T>(), *outputShape);
        if (mode == kSoftmax) {
            SoftmaxGrad<T>(selfOut, pre_grad,  out);
        } else  {
            Logger::Global()->Fatal("OutputGradOp not support other out put.");
        }
        return true;
    }

    template <class T, class Context>
    void OutputGradOp<T, Context>::AsyncRun() {
        if (Context::mode == RunMode::kCpu) {
            Run();
        } else {
            if (!RunOnDevice()) {
                Run();
            }
        }
    }

    template <class T, class Context>
    bool OutputGradOp<T, Context>::RunOnDevice() {
        return false;
    }

    template <class T, class Context>
    OutputGradOp<T, Context>::~OutputGradOp() {

    }



    void OutputGradOpProp::InferShape(std::vector<Shape*> &inShape, Shape *outShape) {
        assert(outShape != nullptr);
        outShape->reShape(*inShape[0]);
    }

    INIT_OPERATOR_PROPERTY_CREATE(OutputGradOpProp, OutputGradOp, true);

}