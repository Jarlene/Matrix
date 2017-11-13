//
// Created by Jarlene on 2017/8/9.
//

#include "matrix/include/op/PredictionOp.h"

namespace matrix {

    template <class T, class Context>
    PredictionOp<T, Context>::PredictionOp(Parameter &param) {
        INIT_PARAMS
    }

    template <class T, class Context>
    bool PredictionOp<T, Context>::Run() {
        return PredictionOp::Run();
    }

    template <class T, class Context>
    void PredictionOp<T, Context>::AsyncRun() {
        Operator::AsyncRun();
    }

    template <class T, class Context>
    bool PredictionOp<T, Context>::RunOnDevice() {
        return false;
    }

    template <class T, class Context>
    PredictionOp<T, Context>::~PredictionOp() {

    }




    PredictionOpProp::PredictionOpProp() {
        param = new Parameter(kFloat);
    }

    PredictionOpProp::PredictionOpProp(const MatrixType &type) {
        param = new Parameter(type);
    }

    PredictionOpProp::~PredictionOpProp() {
        delete param;
    }

    void PredictionOpProp::InferShape(std::vector<Shape*> &inShape, Shape* outShape) {
        outShape->Append(1);
    }

    Operator *PredictionOpProp::CreateOperator(Context context, std::vector<void *> &input, void *output,
                                               std::vector<Shape *> &inShape, Shape *outShape,
                                               std::map<std::string, Any> &args) {
        param->args = &args;
        param->inputs = input;
        param->output = output;
        InferShape(inShape, outShape);
        param->inputShapes = inShape;
        param->outShape = outShape;
        CREATE_OPERATOR(param, PredictionOp, {
            memorySize = sizeof(DType) * param->outShape->Size();
        })
    }


}