//
// Created by Jarlene on 2017/11/5.
//

#include "matrix/include/op/Convolution1DGradOp.h"


namespace matrix {

    template<class T, class xpu>
    Convolution1DGradOp<T, xpu>::Convolution1DGradOp(Parameter &param) {
        INIT_PARAMS

    }

    template<class T, class Context>
    bool Convolution1DGradOp<T, Context>::Run() {
        return true;
    }


    template<class T, class Context>
    void Convolution1DGradOp<T, Context>::AsyncRun() {
        if (Context::mode == RunMode::kCpu) {
            Run();
        } else {
            if (!RunOnDevice()) {
                Run();
            }
        }
    }

    template<class T, class Context>
    Convolution1DGradOp<T, Context>::~Convolution1DGradOp() {

    }

    template<class T, class Context>
    bool Convolution1DGradOp<T, Context>::RunOnDevice() {
        return false;
    }



    void Convolution1DGradOpProp::InferShape(std::vector<Shape *> &inShape, Shape *outShape) {

    }


    INIT_OPERATOR_PROPERTY_CREATE(Convolution1DGradOpProp, Convolution1DGradOp, true);


}