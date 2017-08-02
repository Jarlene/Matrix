//
// Created by Jarlene on 2017/8/2.
//

#include "matrix/include/op/ConvolutionOp.h"


namespace matrix {

    template <class T, class Context>
    ConvolutionOp<T, Context>::ConvolutionOp(ConvolutionParam &param) {

    }

    template <class T, class Context>
    bool ConvolutionOp<T, Context>::Run() {
        return Operator::Run();
    }


    template <class T, class Context>
    void ConvolutionOp<T, Context>::AsyncRun() {
        Operator::AsyncRun();
    }

    template <class T, class Context>
    ConvolutionOp<T, Context>::~ConvolutionOp() {

    }

    template <class T, class Context>
    bool ConvolutionOp<T, Context>::RunOnDevice() {
        return false;
    }

    template <>
    Operator* CreateOp<cpu>(ConvolutionParam param, MatrixType type, std::vector<Shape> &in, std::vector<Shape> out) {
        Operator *op = nullptr;
        TYPE_SWITCH(type, DType, {
            op = new ConvolutionOp<DType, cpu>(param);
        })
        return op;
    }
}