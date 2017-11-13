//
// Created by Jarlene on 2017/11/9.
//

#include "matrix/include/op/RNNOp.h"

namespace matrix {


    template <class T, class Context>
    RNNOp<T, Context>::RNNOp(Parameter &param) {
        INIT_PARAMS
    }

    template <class T, class Context>
    bool RNNOp<T, Context>::Run() {
        return Operator::Run();
    }

    template <class T, class Context>
    void RNNOp<T, Context>::AsyncRun() {
        Operator::AsyncRun();
    }

    template <class T, class Context>
    bool RNNOp<T, Context>::RunOnDevice() {
        return false;
    }

    template <class T, class Context>
    RNNOp<T, Context>::~RNNOp() {

    }


    void RNNOpProp::InferShape(std::vector<Shape*> &inShape, Shape *outShape) {

    }

    INIT_OPERATOR_PROPERTY_CREATE(RNNOpProp, RNNOp, true);

}
