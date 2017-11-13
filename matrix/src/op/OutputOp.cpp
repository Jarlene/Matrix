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
        LossMode outModel = GetArgValue<LossMode>("type", kCrossEntropy);
        Tensor<T> data(Input<T>(DATA), *inputShapes[DATA]);
        Tensor<T> label(Input<T>(LABEL), *inputShapes[LABEL]);
        Tensor<T> out(Output<T>(), *outputShape);
        if (outModel == LossMode::kCrossEntropy) {
            CrossEntropy<T>(data,label, out);
        } else if (outModel == kMSE) {
            RMSLoss<T>(data, label, out);
        } else{
            Logger::Global()->Fatal("OutputOp not support other output.\n");
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





    OutputOpProp::OutputOpProp() {
        param = new Parameter(kFloat);
    }

    OutputOpProp::OutputOpProp(const MatrixType &type) {
        param = new Parameter(type);
    }

    OutputOpProp::~OutputOpProp() {
        delete param;
    }

    void OutputOpProp::InferShape(std::vector<Shape*> &inShape, Shape *outShape) {
        assert(outShape != nullptr);
        outShape->reShape(*inShape[0]);
    }

    Operator *OutputOpProp::CreateOperator(Context context, std::vector<void *> &input, void *output,
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
        CREATE_OPERATOR(context, param, OutputOp, {
            memorySize = sizeof(DType) * param->outShape->Size();
        })
    }



}