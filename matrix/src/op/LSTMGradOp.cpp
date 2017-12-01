//
// Created by Jarlene on 2017/11/24.
//

#include "matrix/include/op/LSTMGradOp.h"

namespace matrix {


    template <class T, class Context>
    LSTMGradOp<T, Context>::LSTMGradOp(Parameter &param) {
        INIT_PARAMS
    }

    template <class T, class Context>
    bool LSTMGradOp<T, Context>::Run() {

        return true;
    }

    template <class T, class Context>
    void LSTMGradOp<T, Context>::AsyncRun() {
        if (Context::mode == RunMode::kCpu) {
            Run();
        } else {
            if (!RunOnDevice()) {
                Run();
            }
        }
    }

    template <class T, class Context>
    bool LSTMGradOp<T, Context>::RunOnDevice() {
        return false;
    }

    template <class T, class Context>
    LSTMGradOp<T, Context>::~LSTMGradOp() {

    }




    void LSTMGradOpProp::InferShape(std::vector<Shape*> &inShape, Shape *outShape) {
        assert(outShape != nullptr);
        if(!param->args->count("hide_num")) {
            Logger::Global()->Fatal("LSTMOpProp InferShape==> need hide_num for out put");
        }
        int hide_num = get<int>(param->args->at("hide_num"));
        outShape->reShape(ShapeN(inShape.at(0)->At(0), hide_num));
    }

    INIT_OPERATOR_PROPERTY_CREATE(LSTMGradOpProp, LSTMGradOp, true);

}