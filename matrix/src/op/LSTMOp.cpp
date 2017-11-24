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
        assert(outShape != nullptr);
        if(!param->args->count("hide_num")) {
            Logger::Global()->Fatal("LSTMOpProp InferShape==> need hide_num for out put");
        }
        int hide_num = get<int>(param->args->at("hide_num"));
        outShape->reShape(ShapeN(inShape.at(0)->At(0), hide_num * 4));
    }

    INIT_OPERATOR_PROPERTY_CREATE(LSTMOpProp, LSTMOp, true);

}