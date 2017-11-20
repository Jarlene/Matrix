//
// Created by Jarlene on 2017/11/20.
//

#include "matrix/include/op/ConcatOp.h"


namespace matrix {

    template <class T, class Context>
    ConcatOp<T, Context>::ConcatOp(Parameter &param) {
        INIT_PARAMS
    }

    template <class T, class Context>
    bool ConcatOp<T, Context>::Run() {
        T *out = Output<T>();
        for (int i = 0; i < InputSize(); ++i) {

        }

        return true;
    }

    template <class T, class Context>
    void ConcatOp<T, Context>::AsyncRun() {
        if (Context::mode == RunMode::kCpu) {
            Run();
        } else {
            if (!RunOnDevice()) {
                Run();
            }
        }
    }

    template <class T, class Context>
    bool ConcatOp<T, Context>::RunOnDevice() {
        return false;
    }

    template <class T, class Context>
    ConcatOp<T, Context>::~ConcatOp() {

    }



    void ConcatOpProp::InferShape(std::vector<Shape*> &inShape, Shape* outShape) {
        assert(outShape != nullptr);
        for (int i = 0; i < inShape.size(); ++i) {
            outShape->Append(inShape.at(i)->Size());
        }

    }

    INIT_OPERATOR_PROPERTY_CREATE(ConcatOpProp, ConcatOp, true);


}