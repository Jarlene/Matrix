//
// Created by Jarlene on 2017/9/6.
//
#include "matrix/include/op/NaiveConvolutionOp.h"
namespace matrix {


    template<class T, class xpu>
    NaiveConvolutionOp<T, xpu>::NaiveConvolutionOp(Parameter &param) {
        INIT_PARAMS
    }

    template<class T, class xpu>
    bool NaiveConvolutionOp<T, xpu>::Run() {
        if (InputSize() <= 2) {

        } else if (InputSize() == 3) {

        } else if (InputSize() == 4) {

        }

        return true;
    }


    template<class T, class xpu>
    void NaiveConvolutionOp<T, xpu>::AsyncRun() {
        if (xpu::mode == RunMode::kCpu) {
            Run();
        } else {
            if (!RunOnDevice()) {
                Run();
            }
        }
    }

    template<class T, class xpu>
    NaiveConvolutionOp<T, xpu>::~NaiveConvolutionOp() {

    }

    template<class T, class xpu>
    bool NaiveConvolutionOp<T, xpu>::RunOnDevice() {
        return false;
    }



    void NaiveConvolutionOpProp::InferShape(std::vector<Shape *> &inShape, Shape *outShape) {

        Shape padding = param->GetArgValue<Shape>("padding", ShapeN(0, 0));
        Shape stride = param->GetArgValue<Shape>("stride",ShapeN(1, 1));
        Shape dilate = param->GetArgValue<Shape>("dilate",ShapeN(1, 1));
        auto order = param->GetArgValue<ImageOrder>("order",NCHW) ;
        int filter_num = 1;
        Shape in = *inShape[0];
        Shape kernel = *inShape[1];
        Shape out = *outShape;
        int n = in[0];
        if (order == NCHW) {
            int channel = in[1];
            int height = (in[2] + padding[0] - (dilate[0] * (kernel[0] - 1) + 1)) / stride[0] + 1;
            int width = (in[3] + padding[1] - (dilate[1] * (kernel[1] - 1) + 1)) / stride[1] + 1;
            filter_num = param->GetArgValue("filter_num", channel);
            out.reShape(ShapeN(n, filter_num, height, width));
        } else {
            int height = (in[1] + padding[0] - (dilate[0] * (kernel[0] - 1) + 1)) / stride[0] + 1;
            int width = (in[2] + padding[1] - (dilate[1] * (kernel[1] - 1) + 1)) / stride[1] + 1;
            int channel = in[3];
            filter_num = param->GetArgValue("filter_num", channel);
            out.reShape(ShapeN(n, height, width, filter_num));
        }

    }

    INIT_OPERATOR_PROPERTY_CREATE(NaiveConvolutionOpProp, NaiveConvolutionOp, true);

}