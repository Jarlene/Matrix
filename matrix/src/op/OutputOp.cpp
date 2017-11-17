//
// Created by Jarlene on 2017/8/9.
//

#include "matrix/include/op/OutputOp.h"

namespace matrix {
    template <class T, class Context>
    OutputOp<T, Context>::OutputOp(Parameter &param) {
        INIT_PARAMS
    }

    template <class T, class Context>
    bool OutputOp<T, Context>::Run() {
        auto outModel = GetArgValue<OutputMode>("type", kSoftmax);
        Tensor<T> data(Input<T>(DATA), *inputShapes->at(DATA));
        Tensor<T> out(Output<T>(), *outputShape);
        if (outModel == kSoftmax) {
            Softmax<T>(data, out);
        } else if (outModel == kSig){
            Sigmoid(data, out);
        } else {
            Logger::Global()->Fatal("OutputOp not support other output.");
        }
        return true;
    }

    template <class T, class Context>
    void OutputOp<T, Context>::AsyncRun() {
        if (Context::mode == RunMode::kCpu) {
            Run();
        } else {
            if (!RunOnDevice()) {
                Run();
            }
        }
    }

    template <class T, class Context>
    bool OutputOp<T, Context>::RunOnDevice() {
        return false;
    }

    template <class T, class Context>
    OutputOp<T, Context>::~OutputOp() {

    }



    void OutputOpProp::InferShape(std::vector<Shape*> &inShape, Shape *outShape) {
        assert(outShape != nullptr);
        outShape->reShape(*inShape[0]);
    }
    INIT_OPERATOR_PROPERTY_CREATE(OutputOpProp, OutputOp, true);


}