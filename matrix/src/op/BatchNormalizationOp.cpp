//
// Created by Jarlene on 2017/11/27.
//

#include "matrix/include/op/BatchNormalizationOp.h"


namespace matrix {

    template <class T, class Context>
    BatchNormalizationOp<T, Context>::BatchNormalizationOp(Parameter &param) {
        INIT_PARAMS
    }

    template <class T, class Context>
    bool BatchNormalizationOp<T, Context>::Run() {

        return true;
    }

    template <class T, class Context>
    void BatchNormalizationOp<T, Context>::AsyncRun() {
        if (Context::mode == RunMode::kCpu) {
            Run();
        } else {
            if (!RunOnDevice()) {
                Run();
            }
        }
    }

    template <class T, class Context>
    bool BatchNormalizationOp<T, Context>::RunOnDevice() {
        return false;
    }

    template <class T, class Context>
    BatchNormalizationOp<T, Context>::~BatchNormalizationOp() {

    }

    void BatchNormalizationOpProp::InferShape(std::vector<Shape*> &inShape, Shape *outShape) {
        assert(outShape != nullptr);
        if(!param->args->count("hide_num")) {
            Logger::Global()->Fatal("LSTMOpProp InferShape==> need hide_num for out put");
        }
        int hide_num = get<int>(param->args->at("hide_num"));
        outShape->reShape(ShapeN(inShape.at(0)->At(0), hide_num));
    }

    INIT_OPERATOR_PROPERTY_CREATE(BatchNormalizationOpProp, BatchNormalizationOp, true);

}