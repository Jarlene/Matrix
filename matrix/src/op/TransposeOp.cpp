//
// Created by Jarlene on 2017/11/21.
//

#include "matrix/include/op/TransposeOp.h"



namespace matrix {

    template <class T, class xpu>
    TransposeOp<T, xpu>::TransposeOp(Parameter &param) {
        INIT_PARAMS
    }

    template <class T, class xpu>
    bool TransposeOp<T, xpu>::Run() {
        T *out = Output<T>();
        for (int i = 0; i < InputSize(); ++i) {
            memcpy(out + i * inputShapes->at(i)->Size(), Input<T>(i), inputShapes->at(i)->Size());
        }
        return true;
    }

    template <class T, class xpu>
    void TransposeOp<T, xpu>::AsyncRun() {
        if (xpu::mode == RunMode::kCpu) {
            Run();
        } else {
            if (!RunOnDevice()) {
                Run();
            }
        }
    }

    template <class T, class xpu>
    bool TransposeOp<T, xpu>::RunOnDevice() {
        return false;
    }

    template <class T, class xpu>
    TransposeOp<T, xpu>::~TransposeOp() {

    }



    void TransposeOpProp::InferShape(std::vector<Shape*> &inShape, Shape* outShape) {
        assert(outShape != nullptr);
        for (int i = 0; i < inShape.size(); ++i) {
            outShape->Append(inShape.at(i)->Size());
        }

    }

    INIT_OPERATOR_PROPERTY_CREATE(TransposeOpProp, TransposeOp, true);


}