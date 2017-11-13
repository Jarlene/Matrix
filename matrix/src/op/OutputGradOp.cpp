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



    OutputGradOpProp::OutputGradOpProp() {
        param = new Parameter(kFloat);
    }

    OutputGradOpProp::OutputGradOpProp(const MatrixType &type) {
        param = new Parameter(type);
    }

    OutputGradOpProp::~OutputGradOpProp() {
        delete param;
    }

    void OutputGradOpProp::InferShape(std::vector<Shape*> &inShape, Shape *outShape) {
        assert(outShape != nullptr);
        outShape->reShape(*inShape[0]);
    }

    Operator *OutputGradOpProp::CreateOperator(Context context, std::vector<void *> &input, void *output,
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
        CREATE_OPERATOR(context, param, OutputGradOp, {
            memorySize = sizeof(DType) * param->outShape->Size();
        })
    }

}