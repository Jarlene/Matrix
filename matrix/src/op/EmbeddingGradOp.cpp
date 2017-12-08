//
// Created by Jarlene on 2017/12/8.
//

#include "matrix/include/op/EmbeddingGradOp.h"



namespace matrix {


    template <class T, class xpu>
    EmbeddingGradOp<T, xpu>::EmbeddingGradOp(Parameter &param) {
        INIT_PARAMS
    }

    template <class T, class xpu>
    bool EmbeddingGradOp<T, xpu>::Run() {
        return true;
    }

    template <class T, class xpu>
    void EmbeddingGradOp<T, xpu>::AsyncRun() {
        if (xpu::mode == RunMode::kCpu) {
            Run();
        } else {
            if (!RunOnDevice()) {
                Run();
            }
        }
    }

    template <class T, class xpu>
    bool EmbeddingGradOp<T, xpu>::RunOnDevice() {
        return false;
    }

    template <class T, class xpu>
    EmbeddingGradOp<T, xpu>::~EmbeddingGradOp() {

    }





    void EmbeddingGradOpProp::InferShape(std::vector<Shape *> &inShape, Shape *outShape) {
        outShape->reShape(ShapeN(inShape[0]->At(0), inShape[0]->StrideExclude(0)));
    }

    INIT_OPERATOR_PROPERTY_CREATE(EmbeddingGradOpProp, EmbeddingGradOp, true);

}
