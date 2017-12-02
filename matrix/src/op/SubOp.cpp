//
// Created by Jarlene on 2017/8/2.
//

#include "matrix/include/op/SubOp.h"

namespace matrix {
    template <class T, class xpu>
    SubOp<T, xpu>::SubOp(Parameter &param) {
        INIT_PARAMS
    }

    template <class T, class xpu>
    bool SubOp<T, xpu>::Run() {
        Tensor<T> in1(Input<T>(INPUT1), *inputShapes->at(INPUT1));
        Tensor<T> in2(Input<T>(INPUT2), *inputShapes->at(INPUT2));
        Tensor<T> out(Output<T>(), *outputShape);
        Sub(in1, in2, out);
        return true;
    }

    template <class T, class xpu>
    void SubOp<T, xpu>::AsyncRun() {
        if (xpu::mode == RunMode::kCpu) {
            Run();
        } else {
            if (!RunOnDevice()) {
                Run();
            }
        }
    }

    template <class T, class xpu>
    bool SubOp<T, xpu>::RunOnDevice() {
        return false;
    }

    template <class T, class xpu>
    SubOp<T, xpu>::~SubOp() {

    }



    void SubOpProp::InferShape(std::vector<Shape*> &inShape, Shape* outShape) {
        assert(outShape != nullptr);
        outShape->reShape(*inShape.at(0));
    }


    INIT_OPERATOR_PROPERTY_CREATE(SubOpProp, SubOp, true);

}