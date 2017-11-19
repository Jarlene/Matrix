//
// Created by Jarlene on 2017/8/2.
//

#include "matrix/include/op/LossOp.h"


namespace matrix {

    template <class T, class Context>
    LossOp<T, Context>::LossOp(Parameter &param) {
        INIT_PARAMS
    }

    template <class T, class Context>
    bool LossOp<T, Context>::Run() {
        auto lossModel = GetArgValue<LossMode>("type", LossMode::kCrossEntropy);
        Tensor<T> data(Input<T>(DATA), *inputShapes->at(DATA));
        Tensor<T> label(Input<T>(LABEL), *inputShapes->at(LABEL));
        Tensor<T> out(Output<T>(), *outputShape);
        Value(out, T(0));
        if (lossModel == LossMode::kCrossEntropy) {
            CrossEntropy<T>(data, label, out);
        } else if (lossModel == LossMode::kMSE) {
            RMSLoss<T>(data, label, out);
        } else if (lossModel == kLikelihood) {

        } else if (lossModel == kSoftmaxCrossEntropy) {
            SoftmaxCrossEntropy(data, label, out);
        } else {
            Logger::Global()->Fatal("LossOp not support other loss.\n");
        }
        return true;
    }

    template <class T, class Context>
    void LossOp<T, Context>::AsyncRun() {
        if (Context::mode == RunMode::kCpu) {
            Run();
        } else {
            if (!RunOnDevice()) {
                Run();
            }
        }
    }

    template <class T, class Context>
    LossOp<T, Context>::~LossOp() {

    }

    template <class T, class Context>
    bool LossOp<T, Context>::RunOnDevice() {
        return false;
    }


    void LossOpProp::InferShape(std::vector<Shape*> &inShape, Shape *outShape) {
        assert(outShape != nullptr);
        outShape->Append(1);
    }

    INIT_OPERATOR_PROPERTY_CREATE(LossOpProp, LossOp, true);

}