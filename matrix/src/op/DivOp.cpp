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
        Tensor<T> in1(Input<T>(INPUT1), *inputShapes->at(INPUT1));
        Tensor<T> in2(Input<T>(INPUT2), *inputShapes->at(INPUT2));
        Tensor<T> out(Output<T>(), *outputShape);
        Div(in1, in2, out);
        return true;
    }

    template <class T, class Context>
    void DivOp<T, Context>::AsyncRun() {
        if (Context::mode == RunMode::kCpu) {
            Run();
        } else {
            if (!RunOnDevice()) {
                Run();
            }
        }
    }

    template <class T, class Context>
    DivOp<T, Context>::~DivOp() {

    }

    template <class T, class Context>
    bool DivOp<T, Context>::RunOnDevice() {
        return false;
    }




    void DivOpProp::InferShape(std::vector<Shape*> &inShape, Shape *outShape) {
        assert(inShape.size() >= 2);
        assert(outShape != nullptr);
        ProduceMulOpShape(inShape, outShape);
    }


    INIT_OPERATOR_PROPERTY_CREATE(DivOpProp, DivOp, true);

}