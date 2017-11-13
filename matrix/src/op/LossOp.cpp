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
        auto lossModel = LossMode::kCrossEntropy;
        if (HasArg("type")) {
            lossModel = GetArgValue<LossMode>("type");
        }
        Tensor<T> data(Input<T>(DATA), *inputShapes[DATA]);
        Tensor<T> label(Input<T>(LABEL), *inputShapes[LABEL]);
        Tensor<T> out(Output<T>(), *outputShape);
        if (lossModel == LossMode::kCrossEntropy) {
            CrossEntropy<T>(data, label, out);
        } else if (lossModel == LossMode::kMSE) {
            RMSLoss<T>(data, label, out);
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






    LossOpProp::LossOpProp() {
        param = new Parameter(kFloat);
    }

    LossOpProp::LossOpProp(const MatrixType &type) {
        param = new Parameter(type);
    }

    LossOpProp::~LossOpProp() {
        delete param;
    }

    void LossOpProp::InferShape(std::vector<Shape*> &inShape, Shape *outShape) {
        assert(outShape != nullptr);
        outShape->Append(1);
    }

    Operator *LossOpProp::CreateOperator(Context context, std::vector<void *> &input, void *output,
                                         std::vector<Shape *> &inShape, Shape *outShape,
                                         std::map<std::string, Any> &args) {
        param->args = &args;
        param->output = output;
        InferShape(inShape, outShape);
        for(auto it = inShape.begin(); it != inShape.end(); ++it) {
            param->inputShapes.push_back(*it);
        }
        for(auto it = input.begin(); it != input.end(); ++it) {
            param->inputs.push_back(*it);
        }
        param->outShape = outShape;
        CREATE_OPERATOR(context, param, LossOp, {
            memorySize = sizeof(DType) * param->outShape->Size();
        })
    }


}