//
// Created by Jarlene on 2017/9/6.
//

#include "matrix/include/op/Convolution1DOp.h"


namespace matrix {

    template<class T, class xpu>
    Convolution1DOp<T, xpu>::Convolution1DOp(Parameter &param) {
        INIT_PARAMS
    }

    template<class T, class xpu>
    bool Convolution1DOp<T, xpu>::Run() {
        int num = inputShapes->at(DATA)->At(0);
        int filterNum = GetArgValue<int>("filter_num");
        const T *inputData = Input<T>(DATA);
        const T *kernelData = Input<T>(KERNEL);
        T *outputData = Output<T>();


        ImageOrder order = NCHW;
        if (HasArg("order")) {
            order = GetArgValue<ImageOrder>("order");
        }

        int channel = 1;
        if (order == NCHW) {
            channel = inputShapes->at(DATA)->At(1);
        } else {
            channel = inputShapes->at(DATA)->At(3);
        }

        Shape kernel = *inputShapes->at(KERNEL);
        Shape stride = GetArgValue<Shape>("stride");
        Shape padding = GetArgValue<Shape>("padding");
        Shape dilate = GetArgValue<Shape>("dilate");

        NaiveConv<T>(inputData, num, channel,
                     inputShapes->at(DATA)->At(2), inputShapes->at(DATA)->At(3),
                     stride[0], stride[1],
                     padding[0], padding[1],
                     kernel[2], kernel[3],
                     dilate[0], dilate[1],
                     filterNum, kernelData, outputData);

        if (InputSize() == 3) {
            Tensor<T> out(Output<T>(),*outputShape);
            Tensor<T> bias(Input<T>(BIAS),*inputShapes->at(BIAS));
            Add<T>(out, bias, out);
        } else {
            Logger::Global()->Fatal("ConvolutionOp do not support other inputs\n");
        }
        return true;
    }


    template<class T, class xpu>
    void Convolution1DOp<T, xpu>::AsyncRun() {
        if (xpu::mode == RunMode::kCpu) {
            Run();
        } else {
            if (!RunOnDevice()) {
                Run();
            }
        }
    }

    template<class T, class xpu>
    Convolution1DOp<T, xpu>::~Convolution1DOp() {

    }

    template<class T, class xpu>
    bool Convolution1DOp<T, xpu>::RunOnDevice() {
        return false;
    }



    void Convolution1DOpProp::InferShape(std::vector<Shape *> &inShape, Shape *outShape) {

        Shape padding = ShapeN(0, 0);
        Shape stride = ShapeN(1, 1);
        Shape dilate = ShapeN(1, 1);
        if (param->HasArg("padding")) {
            padding.reShape(param->GetArgValue<Shape>("padding"));
        } else {
            param->args->insert(std::pair<std::string, Any>("padding", padding));
        }
        if (param->HasArg("stride")) {
            stride.reShape(param->GetArgValue<Shape>("stride"));
        } else {
            param->args->insert(std::pair<std::string, Any>("stride", stride));
        }
        if (param->HasArg("dilate")) {
            dilate.reShape(param->GetArgValue<Shape>("dilate"));
        } else {
            param->args->insert(std::pair<std::string, Any>("dilate", dilate));
        }
        auto order = param->GetArgValue<ImageOrder>("order", NCHW);
        int filter_num = 1;

        Shape in = *inShape[0];
        Shape kernel = *inShape[1];
        Shape out = *outShape;
        Shape colBuffer;
        int n = in[0];
        if (order == NCHW) {
            int channel = in[1];
            int height = (in[2] + 2 * padding[0] - (dilate[0] * (kernel[0] - 1) + 1)) / stride[0] + 1;
            int width = (in[3] + 2 * padding[1] - (dilate[1] * (kernel[1] - 1) + 1)) / stride[1] + 1;
            filter_num = param->GetArgValue<int>("filter_num", channel);

            outShape->reShape(ShapeN(n, filter_num, height, width));
        } else {
            int height = (in[1] + 2 * padding[0] - (dilate[0] * (kernel[0] - 1) + 1)) / stride[0] + 1;
            int width = (in[2] + 2 * padding[1] - (dilate[1] * (kernel[1] - 1) + 1)) / stride[1] + 1;
            int channel = in[3];
            filter_num = param->GetArgValue<int>("filter_num", channel);
            outShape->reShape(ShapeN(n, height, width, filter_num));
        }
    }
    INIT_OPERATOR_PROPERTY_CREATE(Convolution1DOpProp, Convolution1DOp, true);


}