//
// Created by Jarlene on 2017/8/2.
//

#include "matrix/include/op/GRUOp.h"

namespace matrix {

    template <class T, class Context>
    GRUOp<T, Context>::GRUOp(Parameter &param) {
        INIT_PARAMS
    }

    template <class T, class Context>
    bool GRUOp<T, Context>::Run() {
        return Operator::Run();
    }

    template <class T, class Context>
    void GRUOp<T, Context>::AsyncRun() {
        Operator::AsyncRun();
    }

    template <class T, class Context>
    bool GRUOp<T, Context>::RunOnDevice() {
        return false;
    }

    template <class T, class Context>
    GRUOp<T, Context>::~GRUOp() {

    }



    void GRUOpProp::InferShape(std::vector<Shape*> &inShape, Shape *outShape) {

    }

    INIT_OPERATOR_PROPERTY_CREATE(GRUOpProp, GRUOp, true);


}