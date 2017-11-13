//
// Created by Jarlene on 2017/8/2.
//

#include "matrix/include/op/LSTMOp.h"

namespace matrix {

    template <class T, class Context>
    LSTMOp<T, Context>::LSTMOp(Parameter &param) {
        INIT_PARAMS
    }

    template <class T, class Context>
    bool LSTMOp<T, Context>::Run() {


        return true;
    }

    template <class T, class Context>
    void LSTMOp<T, Context>::AsyncRun() {
        if (Context::mode == RunMode::kCpu) {
            Run();
        } else {
            if (!RunOnDevice()) {
                Run();
            }
        }
    }

    template <class T, class Context>
    bool LSTMOp<T, Context>::RunOnDevice() {
        return false;
    }

    template <class T, class Context>
    LSTMOp<T, Context>::~LSTMOp() {

    }



    LSTMOpProp::LSTMOpProp() {
        param = new Parameter(kFloat);
    }

    LSTMOpProp::LSTMOpProp(const MatrixType &type) {
        param = new Parameter(type);
    }

    LSTMOpProp::~LSTMOpProp() {
        delete param;
    }

    void LSTMOpProp::InferShape(std::vector<Shape*> &inShape, Shape *outShape) {

    }

    Operator *LSTMOpProp::CreateOperator(Context context, std::vector<void *> &input, void *output,
                                         std::vector<Shape *> &inShape, Shape *outShape,
                                         std::map<std::string, Any> &args) {
        param->args = &args;
        param->inputs = input;
        param->output = output;
        InferShape(inShape, outShape);
        param->inputShapes = inShape;
        param->outShape = outShape;
        CREATE_OPERATOR(param, LSTMOp, {
            memorySize = sizeof(DType) * param->outShape->Size();
        })
    }


}