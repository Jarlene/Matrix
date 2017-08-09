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
    Operator* CreateOp<CPU>(ConvolutionParam &param) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new ConvolutionOp<DType, CPU>(param);
        })
        return op;
    }

    template <>
    Operator* CreateOp<GPU>(ConvolutionParam &param) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new ConvolutionOp<DType, GPU>(param);
        })
        return op;
    }

    ConvolutionOpProp::ConvolutionOpProp() {
        param = new ConvolutionParam(kFloat);
    }

    ConvolutionOpProp::ConvolutionOpProp(const MatrixType type) {
        param = new ConvolutionParam(type);
    }

    ConvolutionOpProp::~ConvolutionOpProp() {
        delete param;
    }

    void ConvolutionOpProp::InferShape(std::vector<Shape> &inShape, std::vector<Shape> &outShape) {

    }

    Operator *ConvolutionOpProp::CreateOperator(Context context, std::vector<Blob> &input, std::vector<Blob> &output,
                                                std::vector<Shape> &inShape, std::vector<Shape> &outShape) {
        InferShape(inShape, outShape);
        param->inputs = input;
        param->outputs = output;
        param->inputShapes = inShape;
        param->outShapes = outShape;
        BIND_DISPATCH(CreateOp, *param);
    }
}