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
        auto outModel = OutputMode::kSoftmax;
        if (HasArg("type")) {
            outModel = GetArgValue<OutputMode>("type");
        }
        Tensor<T> data(Input<T>(DATA), *inputShapes[DATA]);
        Tensor<T> out(Output<T>(), *outputShape);
        if (outModel == OutputMode::kSoftmax) {
            Softmax<T>(data, out);
        } else {
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
        param->inputs = input;
        param->output = output;
        InferShape(inShape, outShape);
        param->inputShapes = inShape;
        param->outShape = outShape;
        CREATE_OPERATOR(param, OutputOp, {
            memorySize = sizeof(DType) * param->outShape->Size();
        })
    }



}