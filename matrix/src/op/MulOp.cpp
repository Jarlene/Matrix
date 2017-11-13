//
// Created by Jarlene on 2017/8/2.
//

#include "matrix/include/op/MulOp.h"

namespace matrix {


    template <class T, class Context>
    MulOp<T, Context>::MulOp(Parameter &param) {
        INIT_PARAMS
    }

    template <class T, class Context>
    bool MulOp<T, Context>::Run() {
        Tensor<T> data(Input<T>(INPUT1), *inputShapes->at(INPUT1));
        Tensor<T> weight(Input<T>(INPUT2), *inputShapes->at(INPUT2));
        Tensor<T> out(Output<T>(), *outputShape);
        MatrixMul<T>(data, false, weight, false, out);
        return true;
    }

    template <class T, class Context>
    void MulOp<T, Context>::AsyncRun() {
        if (Context::mode == RunMode::kCpu) {
            Run();
        } else {
            if (!RunOnDevice()) {
                Run();
            }
        }
    }

    template <class T, class Context>
    bool MulOp<T, Context>::RunOnDevice() {
        return false;
    }

    template <class T, class Context>
    MulOp<T, Context>::~MulOp() {

    }



    void MulOpProp::InferShape(std::vector<Shape*> &inShape, Shape *outShape) {
        assert(inShape.size() >= 2);
        assert(outShape != nullptr);
        ProduceMulOpShape(inShape, outShape);
    }

    INIT_OPERATOR_PROPERTY_CREATE(MulOpProp, MulOp, true);
}