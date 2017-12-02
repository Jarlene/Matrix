//
// Created by Jarlene on 2017/8/2.
//

#include "matrix/include/op/GRUOp.h"

namespace matrix {

    template <class T, class xpu>
    GRUOp<T, xpu>::GRUOp(Parameter &param) {
        INIT_PARAMS
    }

    template <class T, class xpu>
    bool GRUOp<T, xpu>::Run() {
        return Operator::Run();
    }

    template <class T, class xpu>
    void GRUOp<T, xpu>::AsyncRun() {
        Operator::AsyncRun();
    }

    template <class T, class xpu>
    bool GRUOp<T, xpu>::RunOnDevice() {
        return false;
    }

    template <class T, class xpu>
    GRUOp<T, xpu>::~GRUOp() {

    }



    void GRUOpProp::InferShape(std::vector<Shape*> &inShape, Shape *outShape) {

    }

    INIT_OPERATOR_PROPERTY_CREATE(GRUOpProp, GRUOp, true);


}