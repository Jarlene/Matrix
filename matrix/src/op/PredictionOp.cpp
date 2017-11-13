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



    void PredictionOpProp::InferShape(std::vector<Shape*> &inShape, Shape* outShape) {
        outShape->Append(1);
    }

    INIT_OPERATOR_PROPERTY_CREATE(PredictionOpProp, PredictionOp, true);

}