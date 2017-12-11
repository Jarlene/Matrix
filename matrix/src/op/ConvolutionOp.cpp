//
// Created by Jarlene on 2017/8/2.
//

#include "matrix/include/store/MemoryManager.h"
#include "matrix/include/op/ConvolutionOp.h"


namespace matrix {

    template<class T, class xpu>
    ConvolutionOp<T, xpu>::ConvolutionOp(Parameter &param) {
        INIT_PARAMS

    }

    template<class T, class xpu>
    bool ConvolutionOp<T, xpu>::Run() {

        int num = inputShapes->at(DATA)->At(0);

        ImageOrder order = NCHW;
        if (HasArg("order")) {
            order = GetArgValue<ImageOrder>("order");
        }
        int imageSize = 1;
        int outsize = 1;
        int channel = 1;
        if (order == NCHW) {
            channel = inputShapes->at(DATA)->At(1);
            imageSize = inputShapes->at(DATA)->At(2) * inputShapes->at(DATA)->At(3);
            outsize = outputShape->At(2) * outputShape->At(3);
        } else {
            imageSize = inputShapes->at(DATA)->At(1) * inputShapes->at(DATA)->At(2);
            channel = inputShapes->at(DATA)->At(3);
            outsize = outputShape->At(1) * outputShape->At(2);
        }

        int group = GetArgValue<int>("group", 1);
        int filterNum = GetArgValue<int>("filter_num", channel);

        Shape kernel = *inputShapes->at(KERNEL);
        Shape stride = GetArgValue<Shape>("stride", ShapeN(1, 1));
        Shape padding = GetArgValue<Shape>("padding", ShapeN(0, 0));
        Shape dilate = GetArgValue<Shape>("dilate", ShapeN(1, 1));


        const int input_offset = channel / group * imageSize;

        const int output_offset = outputShape->At(1) * outputShape->At(2) * outputShape->At(3) / group;

        const int filter_offset = kernel.Size() / group;

        const T *inputData = Input<T>(DATA);
        const T *kernelData = Input<T>(KERNEL);
        T *outputData = Output<T>();
        T *colData = InputNonConst<T>(InputSize() - 1);

        for (int i = 0; i < num; ++i) {
            for (int j = 0; j < group; ++j) {
                Img2Col<T>(inputData + j * input_offset, *inputShapes->at(DATA),
                           kernel, stride,
                           padding, dilate,
                           colData, order);
                int M = filterNum / group;
                int N = outsize;
                int K = channel / group * kernel.At(2) * kernel.At(3);
                CPUGemm<T>(NoTrans, NoTrans, M, N, K, T(1.0), kernelData + j * filter_offset, colData,
                           T(0.0), outputData + j * output_offset);
            }
            inputData += input_offset * group;
            outputData += output_offset * group;
        }
        if (InputSize()  == 4) {
            Shape flatten = ShapeN(filterNum, outsize);
            for (int i = 0; i < num; ++i) {
                Tensor<T> ou(Output<T>() + i * flatten.Size(), flatten);
                Tensor<T> bias(Input<T>(BIAS), *inputShapes->at(BIAS));
                Add(ou, bias, ou);
            }
        }
        return true;
    }


    template<class T, class xpu>
    void ConvolutionOp<T, xpu>::AsyncRun() {
        if (xpu::mode == RunMode::kCpu) {
            Run();
        } else {
            if (!RunOnDevice()) {
                Run();
            }
        }
    }

    template<class T, class xpu>
    ConvolutionOp<T, xpu>::~ConvolutionOp() {

    }

    template<class T, class xpu>
    bool ConvolutionOp<T, xpu>::VariableNode(std::function<void(std::initializer_list<Shape *> shapes)> func) {
        if (InputSize() == 1) {
            if (!HasArg("filter")) {
                Logger::Global()->Fatal("ConvolutionOp no filter for input");
            }
            Shape filter = GetArgValue<Shape>("filter");

            ImageOrder order = GetArgValue<ImageOrder>("order", NCHW);
            int channel = 0;
            if (order == NCHW) {
                channel = inputShapes->at(DATA)->At(1);
            } else {
                channel = inputShapes->at(DATA)->At(3);
            }
            int filter_num = GetArgValue<int>("filter_num", channel);
            filter.reShape(ShapeN(filter_num, channel, filter[0], filter[1]));
            if (HasArg("with_bias") && GetArgValue<bool>("with_bias")) {
                Shape bias;
                bias.Append(filter_num);
                func({&filter, &bias});
            } else {
                func({&filter});
            }
            return true;
        }
        return false;
    };

    template<class T, class xpu>
    bool ConvolutionOp<T, xpu>::RunOnDevice() {
        return false;
    }

    template<class T, class xpu>
    bool ConvolutionOp<T, xpu>::ShareNodes(std::function<void(std::initializer_list<Shape *> shapes)> func) {
        bool with_bias = GetArgValue<bool>("with_bias", true);
        if ((InputSize() < 4 && with_bias) || (InputSize() < 3 && !with_bias)) {
            int group = GetArgValue<int>("group", 1);
            if (InputSize() == 1) {
                if (!HasArg("filter")) {
                    Logger::Global()->Fatal("ConvolutionOp no filter for input");
                }
                Shape filter = GetArgValue<Shape>("filter");
                int filter_num = GetArgValue<int>("filter_num");
                ImageOrder order = GetArgValue<ImageOrder>("order", NCHW);
                int channel = 0;
                if (order == NCHW) {
                    channel = inputShapes->at(DATA)->At(1);
                } else {
                    channel = inputShapes->at(DATA)->At(3);
                }
                filter.reShape(ShapeN(filter_num, channel, filter[0], filter[1]));
                auto colShape = ShapeN(channel/group, inputShapes->at(KERNEL)->At(2) , inputShapes->at(KERNEL)->At(3),
                                       outputShape->At(2) , outputShape->At(3));
                func({&colShape});
            } else {
                ImageOrder order = GetArgValue<ImageOrder>("order", NCHW);
                int channel = 0;
                if (order == NCHW) {
                    channel = inputShapes->at(DATA)->At(1);
                } else {
                    channel = inputShapes->at(DATA)->At(3);
                }
                auto colShape = ShapeN(channel/group, inputShapes->at(KERNEL)->At(2) , inputShapes->at(KERNEL)->At(3),
                                       outputShape->At(2) , outputShape->At(3));
                func({&colShape});
            }
            return true;
        }
        return false;
    }


    void ConvolutionOpProp::InferShape(std::vector<Shape *> &inShape, Shape *outShape) {

        if (inShape.size() == 1) {
            return;
        }

        Shape padding = ShapeN(0, 0);
        Shape stride = ShapeN(1, 1);
        Shape dilate = ShapeN(1, 1);
        if (param->args->count("padding")) {
            padding.reShape(get<Shape>(param->args->at("padding")));
        } else {
            param->args->insert(std::pair<std::string, Any>("padding", padding));
        }
        if (param->args->count("stride")) {
            stride.reShape(get<Shape>(param->args->at("stride")));
        } else {
            param->args->insert(std::pair<std::string, Any>("stride", stride));
        }
        if (param->args->count("dilate")) {
            dilate.reShape(get<Shape>(param->args->at("dilate")));
        } else {
            param->args->insert(std::pair<std::string, Any>("dilate", dilate));
        }
        ImageOrder order = NCHW;
        if (param->args->count("order")) {
            order = get<ImageOrder>(param->args->at("order"));
        }


        Shape in = *inShape[0];
        Shape kernel = *inShape[1];

        int n = in[0];
        int kernel_h = kernel[2];
        int kernel_w = kernel[3];
        int filter_num = kernel[0];
        if (order == NCHW) {
            int height = (in[2] + 2 * padding[0] - (dilate[0] * (kernel_h - 1) + 1)) / stride[0] + 1;
            int width = (in[3] + 2 * padding[1] - (dilate[1] * (kernel_w - 1) + 1)) / stride[1] + 1;
            outShape->reShape(ShapeN(n, filter_num, height, width));
        } else {
            int height = (in[1] + 2 * padding[0] - (dilate[0] * (kernel_h - 1) + 1)) / stride[0] + 1;
            int width = (in[2] + 2 * padding[1] - (dilate[1] * (kernel_w - 1) + 1)) / stride[1] + 1;
            outShape->reShape(ShapeN(n, height, width, filter_num));
        }
    }

    INIT_OPERATOR_PROPERTY_CREATE(ConvolutionOpProp, ConvolutionOp, true);


}