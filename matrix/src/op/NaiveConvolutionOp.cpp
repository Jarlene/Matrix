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

        Shape padding = ShapeN(0, 0);
        Shape stride = ShapeN(1, 1);
        Shape dilate = ShapeN(1, 1);
        if (param->args->count("padding")) {
            padding.reShape(get<Shape>(param->args->at("padding")));
        }
        if (param->args->count("stride")) {
            stride.reShape(get<Shape>(param->args->at("stride")));
        }
        if (param->args->count("dilate")) {
            dilate.reShape(get<Shape>(param->args->at("dilate")));
        }
        ImageOrder order = NCHW;
        if (param->args->count("order")) {
            order = get<ImageOrder>(param->args->at("order"));
        }


        int group = 1;
        if (param->args->count("group")) {
            group = get<int>(param->args->at("group"));
        }
        int filter_num = 1;


        Shape in = *inShape[0];
        Shape kernel = *inShape[1];
        Shape out = *outShape;
        Shape colBuffer = *inShape[inShape.size() - 1];
        int n = in[0];
        int kernel_h = kernel[0];
        int kernel_w = kernel[1];
        if (order == NCHW) {
            int channel = in[1];
            int height = (in[2] + padding[0] - (dilate[0] * (kernel[0] - 1) + 1)) / stride[0] + 1;
            int width = (in[3] + padding[1] - (dilate[1] * (kernel[1] - 1) + 1)) / stride[1] + 1;
            filter_num = channel;
            if (param->args->count("filter_num")) {
                filter_num = get<int>(param->args->at("filter_num"));
            }
            out.reShape(ShapeN(n, filter_num, height, width));
            int c = channel / group * kernel.Size();

            colBuffer.reShape(ShapeN(c, height, width));
            kernel.reShape(ShapeN(filter_num, channel, kernel_h, kernel_w));
        } else {
            int height = (in[1] + padding[0] - (dilate[0] * (kernel[0] - 1) + 1)) / stride[0] + 1;
            int width = (in[2] + padding[1] - (dilate[1] * (kernel[1] - 1) + 1)) / stride[1] + 1;
            int channel = in[3];
            filter_num = channel;
            if (param->args->count("filter_num")) {
                filter_num = get<int>(param->args->at("filter_num"));
            }
            out.reShape(ShapeN(n, height, width, filter_num));
            int c = channel / group * kernel.Size();
            colBuffer.reShape(ShapeN(c, height, width));
            kernel.reShape(ShapeN(filter_num, channel, kernel_h, kernel_w));
        }

    }

    INIT_OPERATOR_PROPERTY_CREATE(NaiveConvolutionOpProp, NaiveConvolutionOp, true);

}