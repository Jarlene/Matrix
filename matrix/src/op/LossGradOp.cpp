//
// Created by Jarlene on 2017/11/5.
//

#include "matrix/include/op/LossGradOp.h"



namespace matrix {

    template <class T, class Context>
    LossGradOp<T, Context>::LossGradOp(Parameter &param) {
        INIT_PARAMS
    }

    template <class T, class Context>
    bool LossGradOp<T, Context>::Run() {
        auto lossModel = GetArgValue<LossMode>("type", LossMode::kCrossEntropy);
        Tensor<T> pre(Input<T>(PRE_GRAD), *inputShapes->at(PRE_GRAD));
        Tensor<T> data(Input<T>(DATA), *inputShapes->at(DATA));
        Tensor<T> label(Input<T>(LABEL), *inputShapes->at(LABEL));
        Tensor<T> out(Output<T>(), *outputShape);

        if (lossModel == LossMode::kCrossEntropy) {
            CrossEntropyGrad<T>(data, label, out);
        } else if (lossModel == LossMode::kMSE) {
            RMSLossGrad<T>(data, label, out);
        } else if (lossModel == kSoftmaxCrossEntropy) {
            SoftmaxCrossEntropyGrad(data, label, out);
        } else {
            Logger::Global()->Fatal("LossOp not support other loss.\n");
        }
        Scale<T>(out, pre.Data()[0]/data.Size());
        return true;
    }

    template <class T, class Context>
    void LossGradOp<T, Context>::AsyncRun() {
        if (Context::mode == RunMode::kCpu) {
            Run();
        } else {
            if (!RunOnDevice()) {
                Run();
            }
        }
    }

    template <class T, class Context>
    LossGradOp<T, Context>::~LossGradOp() {

    }

    template <class T, class Context>
    bool LossGradOp<T, Context>::RunOnDevice() {
        return false;
    }



    void LossGradOpProp::InferShape(std::vector<Shape*> &inShape, Shape *outShape) {
        assert(outShape != nullptr);
        int idx = get<int>(param->args->at("input_idx"));
        if (idx == 0) {
            outShape->reShape(*inShape[2]);
        } else if (idx == 1){
            outShape->reShape(*inShape[3]);
        }
    }

    INIT_OPERATOR_PROPERTY_CREATE(LossGradOpProp, LossGradOp, true);

}