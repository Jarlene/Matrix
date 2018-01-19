//
// Created by Jarlene on 2017/11/9.
//

#include "matrix/include/op/RNNOp.h"

namespace matrix {


    template <class T, class xpu>
    RNNOp<T, xpu>::RNNOp(Parameter &param) {
        INIT_PARAMS
    }

    template <class T, class xpu>
    bool RNNOp<T, xpu>::Run() {
        return Operator::Run();
    }

    template <class T, class xpu>
    void RNNOp<T, xpu>::AsyncRun() {
        Operator::AsyncRun();
    }

    template <class T, class xpu>
    bool RNNOp<T, xpu>::RunOnDevice() {
        return false;
    }

    template <class T, class xpu>
    RNNOp<T, xpu>::~RNNOp() {

    }

    template <class T, class xpu>
    bool RNNOp<T, xpu>::VariableNode(std::function<void(std::initializer_list<Shape *> shapes)> func) {
        if (InputSize() == 1) {
            if (!HasArg("hide_num")) {
                Logger::Global()->Fatal("RNNOp need hide num");
            }
            int hideNum = GetArgValue<int>("hide_num");
            int rank = inputShapes->at(0)->Rank();
            Shape w;
            w.reShape(ShapeN(inputShapes->at(0)->At(rank - 1), hideNum));
            if(HasArg("with_bias")) {
                Shape b;
                b.Append(hideNum);
                func({&w, &b});
            } else {
                func({&w});
            }
            return true;
        }
        return false;
    }

    template<class T, class xpu>
    bool RNNOp<T, xpu>::ShareNodes(std::function<void(std::initializer_list<Shape *> shapes)> func) {
        return false;
    }


    void RNNOpProp::InferShape(std::vector<Shape*> &inShape, Shape *outShape) {
        if (inShape.size() == 1) {
            return;
        }






    }

    INIT_OPERATOR_PROPERTY_CREATE(RNNOpProp, RNNOp, true);

    template <class T>
    void RNNState<T>::clear() {
        if(this->shape != nullptr && this->ht != nullptr) {
            Value(shape->Size(), ht, T(0));
        }

    }

}
