//
// Created by Jarlene on 2017/8/2.
//

#include "matrix/include/op/LSTMOp.h"

namespace matrix {

    template <class T, class Context>
    LSTMOp<T, Context>::LSTMOp(Parameter &param) {
        INIT_PARAMS
    }

    template <class T, class Context>
    bool LSTMOp<T, Context>::Run() {


        return true;
    }

    template <class T, class Context>
    void LSTMOp<T, Context>::AsyncRun() {
        if (Context::mode == RunMode::kCpu) {
            Run();
        } else {
            if (!RunOnDevice()) {
                Run();
            }
        }
    }

    template <class T, class Context>
    bool LSTMOp<T, Context>::RunOnDevice() {
        return false;
    }

    template <class T, class Context>
    LSTMOp<T, Context>::~LSTMOp() {

    }

    template <class T, class Context>
    bool LSTMOp<T, Context>::ShareNodes(std::function<void(std::initializer_list<Shape *> shapes)> func) {
        return Operator::ShareNodes(func);
    }

    template <class T, class Context>
    bool LSTMOp<T, Context>::VariableNode(std::function<void(std::initializer_list<Shape *> shapes)> func) {
        return Operator::VariableNode(func);
    }


    void LSTMOpProp::InferShape(std::vector<Shape*> &inShape, Shape *outShape) {

    }

    INIT_OPERATOR_PROPERTY_CREATE(LSTMOpProp, LSTMOp, true);

}