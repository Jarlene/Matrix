//
// Created by Jarlene on 2017/8/9.
//

#include "matrix/include/op/PredictionOp.h"

namespace matrix {

    template <class T, class xpu>
    PredictionOp<T, xpu>::PredictionOp(Parameter &param) {
        INIT_PARAMS
    }

    template <class T, class xpu>
    bool PredictionOp<T, xpu>::Run() {
        return PredictionOp::Run();
    }

    template <class T, class xpu>
    void PredictionOp<T, xpu>::AsyncRun() {
        Operator::AsyncRun();
    }

    template <class T, class xpu>
    bool PredictionOp<T, xpu>::RunOnDevice() {
        return false;
    }

    template <class T, class xpu>
    PredictionOp<T, xpu>::~PredictionOp() {

    }



    void PredictionOpProp::InferShape(std::vector<Shape*> &inShape, Shape* outShape) {
        outShape->Append(1);
    }

    INIT_OPERATOR_PROPERTY_CREATE(PredictionOpProp, PredictionOp, true);

}