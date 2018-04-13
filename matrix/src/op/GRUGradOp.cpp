//
// Created by Jarlene on 2017/8/2.
//

#include "matrix/include/op/GRUGradOp.h"

namespace matrix {

    template <class T, class xpu>
    GRUGradOp<T, xpu>::GRUGradOp(Parameter &param) {
        INIT_PARAMS
    }

    template <class T, class xpu>
    bool GRUGradOp<T, xpu>::Run() {
        return Operator::Run();
    }

    template <class T, class xpu>
    void GRUGradOp<T, xpu>::AsyncRun() {
        Operator::AsyncRun();
    }

    template <class T, class xpu>
    bool GRUGradOp<T, xpu>::RunOnDevice() {
        return false;
    }

    template <class T, class xpu>
    GRUGradOp<T, xpu>::~GRUGradOp() {

    }



    void GRUGradOpProp::InferShape(std::vector<Shape*> &inShape, Shape *outShape) {

    }

    INIT_OPERATOR_PROPERTY_CREATE(GRUGradOpProp, GRUGradOp, true);


}