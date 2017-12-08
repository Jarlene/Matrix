//
// Created by Jarlene on 2017/12/8.
//

#include "matrix/include/op/EmbeddingOp.h"



namespace matrix {


    template <class T, class xpu>
    EmbeddingOp<T, xpu>::EmbeddingOp(Parameter &param) {
        INIT_PARAMS
    }

    template <class T, class xpu>
    bool EmbeddingOp<T, xpu>::Run() {
        return true;
    }

    template <class T, class xpu>
    void EmbeddingOp<T, xpu>::AsyncRun() {
        if (xpu::mode == RunMode::kCpu) {
            Run();
        } else {
            if (!RunOnDevice()) {
                Run();
            }
        }
    }

    template <class T, class xpu>
    bool EmbeddingOp<T, xpu>::RunOnDevice() {
        return false;
    }

    template <class T, class xpu>
    EmbeddingOp<T, xpu>::~EmbeddingOp() {

    }





    void EmbeddingOpProp::InferShape(std::vector<Shape *> &inShape, Shape *outShape) {
        outShape->reShape(ShapeN(inShape[0]->At(0), inShape[0]->StrideExclude(0)));
    }

    INIT_OPERATOR_PROPERTY_CREATE(EmbeddingOpProp, EmbeddingOp, true);

}
