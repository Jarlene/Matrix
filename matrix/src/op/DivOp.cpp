//
// Created by Jarlene on 2017/8/2.
//

#include "matrix/include/op/DivOp.h"

namespace matrix {

    template <class T, class Context>
    DivOp<T, Context>::DivOp(Parameter &param) {
        INIT_PARAMS
    }

    template <class T, class Context>
    bool DivOp<T, Context>::Run() {
        return Operator::Run();
    }

    template <class T, class Context>
    void DivOp<T, Context>::AsyncRun() {
        Operator::AsyncRun();
    }

    template <class T, class Context>
    DivOp<T, Context>::~DivOp() {

    }

    template <class T, class Context>
    bool DivOp<T, Context>::RunOnDevice() {
        return false;
    }




    void DivOpProp::InferShape(std::vector<Shape*> &inShape, Shape *outShape) {
        outShape->reShape(*inShape.at(0));
    }


    INIT_OPERATOR_PROPERTY_CREATE(DivOpProp, DivOp, true);

}