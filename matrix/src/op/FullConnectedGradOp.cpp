//
// Created by Jarlene on 2017/8/23.
//


#include "matrix/include/op/FullConnectedGradOp.h"

namespace matrix {

    template <class T, class Context>
    FullConnectedGradOp<T, Context>::FullConnectedGradOp(Parameter &param) {
        INIT_PARAMS
    }

    template <class T, class Context>
    bool FullConnectedGradOp<T, Context>::Run() {
        if (!HasArg("input_idx")) {
            Logger::Global()->Fatal("FullConnectedGradOp not support. \n");
        }
        int idx = GetArgValue<int>("input_idx");
        Tensor<T> pre_grad(Input<T>(PRE_GRAD), *inputShapes->at(PRE_GRAD));
        Tensor<T> data(Input<T>(DATA), *inputShapes->at(DATA));
        Tensor<T> weight(Input<T>(WEIGHT), *inputShapes->at(WEIGHT));

        Tensor<T> out(Output<T>(), *outputShape);
        switch (idx + 2) {
            case DATA:
                MatrixMul<T>(pre_grad, false, weight, true, out);
                break;
            case WEIGHT:
                MatrixMul<T>(data, true, pre_grad, false, out);
                break;
            case BIAS:
                Copy<T>(pre_grad, out);
                break;
            default:
                Logger::Global()->Fatal("FullConnectedGradOp not support. \n");
                break;
        }
        return true;
    }

    template <class T, class Context>
    void FullConnectedGradOp<T, Context>::AsyncRun() {
        if (Context::mode == RunMode::kCpu) {
            Run();
        } else {
            if (!RunOnDevice()) {
                Run();
            }
        }
    }

    template <class T, class Context>
    FullConnectedGradOp<T, Context>::~FullConnectedGradOp() {

    }

    template <class T, class Context>
    bool FullConnectedGradOp<T, Context>::RunOnDevice() {
        return false;
    }




    FullConnectedGradOpProp::FullConnectedGradOpProp() {
        param = new Parameter(kFloat);
    }

    FullConnectedGradOpProp::FullConnectedGradOpProp(const MatrixType &type) {
        param = new Parameter(type);
    }

    FullConnectedGradOpProp::~FullConnectedGradOpProp() {
        delete param;
    }

    void FullConnectedGradOpProp::InferShape(std::vector<Shape*> &inShape, Shape *outShape) {
        if (param->args->count("input_idx")) {
            int idx = get<int>(param->args->at("input_idx"));
            outShape->reShape(*inShape.at(idx + 2));
        }

    }

    Operator *FullConnectedGradOpProp::CreateOperator(Context context, std::vector<void *> *input, void *output,
                                                      std::vector<Shape *> *inShape, Shape *outShape,
                                                      std::map<std::string, Any> &args) {

        param->args = &args;
        param->output = output;
        InferShape(*inShape, outShape);
        param->inputShapes = inShape;
        param->inputs = input;
        param->outShape = outShape;
        CREATE_OPERATOR(context, param, FullConnectedGradOp, {
            memorySize = sizeof(DType) * param->outShape->Size();
        })
    }



}