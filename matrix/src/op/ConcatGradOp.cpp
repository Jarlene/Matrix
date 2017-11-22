//
// Created by Jarlene on 2017/11/21.
//

#include "matrix/include/op/ConcatGradOp.h"


namespace matrix {
    template <class T, class Context>
    ConcatGradOp<T, Context>::ConcatGradOp(Parameter &param) {
        INIT_PARAMS
    }

    template <class T, class Context>
    bool ConcatGradOp<T, Context>::Run() {
        int idx = GetArgValue<int>("input_idx", -1);
        if(idx == -1) {
            Logger::Global()->Fatal("ConcatGradOp need input idx");
        }
        T *out = Output<T>();
        T *pre_grad = Input<T>(0);
        int size = inputShapes->at(idx)->Size();
        memcpy(out, pre_grad + idx * size, size);
        return true;
    }

    template <class T, class Context>
    void ConcatGradOp<T, Context>::AsyncRun() {
        if (Context::mode == RunMode::kCpu) {
            Run();
        } else {
            if (!RunOnDevice()) {
                Run();
            }
        }
    }

    template <class T, class Context>
    bool ConcatGradOp<T, Context>::RunOnDevice() {
        return false;
    }

    template <class T, class Context>
    ConcatGradOp<T, Context>::~ConcatGradOp() {

    }



    void ConcatGradOpProp::InferShape(std::vector<Shape*> &inShape, Shape* outShape) {
        assert(outShape != nullptr);
        if (!param->args->count("input_idx")) {
            Logger::Global()->Fatal("ConcatGradOpProp need input idx");
        }
        int idx = get<int>(param->args->at("input_idx"));
        outShape->reShape(*inShape.at(idx));

    }

    INIT_OPERATOR_PROPERTY_CREATE(ConcatGradOpProp, ConcatGradOp, true);

}