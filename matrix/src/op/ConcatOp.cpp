//
// Created by Jarlene on 2017/11/20.
//

#include "matrix/include/op/ConcatOp.h"


namespace matrix {

    template <class T, class xpu>
    ConcatOp<T, xpu>::ConcatOp(Parameter &param) {
        INIT_PARAMS
    }

    template <class T, class xpu>
    bool ConcatOp<T, xpu>::Run() {
        T *out = Output<T>();
        int total = 0;
        for (int i = 0; i < InputSize(); ++i) {
            int size = inputShapes->at(i)->Size();
            CPUCopy(size, Input<T>(i), 1, out + total, 1);
            total += size;
        }
        return true;
    }

    template <class T, class xpu>
    void ConcatOp<T, xpu>::AsyncRun() {
        if (xpu::mode == RunMode::kCpu) {
            Run();
        } else {
            if (!RunOnDevice()) {
                Run();
            }
        }
    }

    template <class T, class xpu>
    bool ConcatOp<T, xpu>::RunOnDevice() {
        return false;
    }

    template <class T, class xpu>
    ConcatOp<T, xpu>::~ConcatOp() {

    }



    void ConcatOpProp::InferShape(std::vector<Shape*> &inShape, Shape* outShape) {
        assert(outShape != nullptr);
        int batch = inShape.at(0)->At(0);
        outShape->Append(batch);
        int total = 0;
        for (int i = 0; i < inShape.size(); ++i) {
            assert(batch == inShape.at(i)->At(0));
            total += inShape.at(i)->Size()/batch;
        }
        outShape->Append(total);

    }

    INIT_OPERATOR_PROPERTY_CREATE(ConcatOpProp, ConcatOp, true);


}