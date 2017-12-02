//
// Created by Jarlene on 2017/8/9.
//

#include "matrix/include/op/OutputOp.h"

namespace matrix {
    template <class T, class xpu>
    OutputOp<T, xpu>::OutputOp(Parameter &param) {
        INIT_PARAMS
    }

    template <class T, class xpu>
    bool OutputOp<T, xpu>::Run() {
        auto outModel = GetArgValue<OutputMode>("type", kSoftmax);
        Tensor<T> data(Input<T>(DATA), *inputShapes->at(DATA));
        Tensor<T> out(Output<T>(), *outputShape);
        if (outModel == kSoftmax) {
            Softmax<T>(data, out);
        } else {
            Logger::Global()->Fatal("OutputOp not support other output.");
        }
        return true;
    }

    template <class T, class xpu>
    void OutputOp<T, xpu>::AsyncRun() {
        if (xpu::mode == RunMode::kCpu) {
            Run();
        } else {
            if (!RunOnDevice()) {
                Run();
            }
        }
    }

    template <class T, class xpu>
    bool OutputOp<T, xpu>::RunOnDevice() {
        return false;
    }

    template <class T, class xpu>
    OutputOp<T, xpu>::~OutputOp() {

    }



    void OutputOpProp::InferShape(std::vector<Shape*> &inShape, Shape *outShape) {
        assert(outShape != nullptr);
        outShape->reShape(*inShape[0]);
    }
    INIT_OPERATOR_PROPERTY_CREATE(OutputOpProp, OutputOp, true);


}