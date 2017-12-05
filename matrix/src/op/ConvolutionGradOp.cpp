//
// Created by Jarlene on 2017/9/8.
//

#include "matrix/include/op/ReduceOp.h"
#include "matrix/include/store/MemoryManager.h"
#include "matrix/include/op/ConvolutionGradOp.h"


namespace matrix {

    template<class T, class xpu>
    ConvolutionGradOp<T, xpu>::ConvolutionGradOp(Parameter &param) {
        INIT_PARAMS

    }

    template<class T, class xpu>
    bool ConvolutionGradOp<T, xpu>::Run() {
        ImageOrder order = GetArgValue<ImageOrder>("order", NCHW);
        int num = inputShapes->at(DATA)->At(0);
        
        int channel = 1;
        int outSize = 1;
        int input_width = 0, input_height = 0;
        if (order == NCHW) {
            channel = inputShapes->at(DATA)->At(1);
            input_width = inputShapes->at(DATA)->At(2);
            input_height = inputShapes->at(DATA)->At(3);
            outSize = inputShapes->at(SELF_OUT)->At(2) * inputShapes->at(SELF_OUT)->At(3);
        } else {
            input_width = inputShapes->at(DATA)->At(1);
            input_height = inputShapes->at(DATA)->At(2);
            channel = inputShapes->at(DATA)->At(3);
            outSize = inputShapes->at(SELF_OUT)->At(1) * inputShapes->at(SELF_OUT)->At(2);
        }

        int filterNum = GetArgValue<int>("filter_num", channel);
        int group = GetArgValue<int>("group", 1);

        Shape kernel = *inputShapes->at(KERNEL);
        Shape stride = GetArgValue<Shape>("stride", ShapeN(1, 1));
        Shape padding = GetArgValue<Shape>("padding", ShapeN(0, 0));
        Shape dilate = GetArgValue<Shape>("dilate", ShapeN(1, 1));


        int inputOffSize = channel / group * input_width * input_height;;
        int outputOffset = filterNum / group * outSize;
        int filterOffset = kernel.Size() / group;

        const T *preGrad = Input<T>(PRE_GRAG);
        T *out = Output<T>();
        int index = GetArgValue<int>("input_idx", -1);

        T *colData = InputNonConst<T>(InputSize() - 1);

        if (index == 0) {
            const T *filterData = Input<T>(KERNEL);
            int K = filterNum / group;
            int N = outSize;
            int M = channel / group * kernel[2] * kernel[3];

            for (int i = 0; i < num; ++i) {
                for (int j = 0; j < group; ++j) {
                    CPUGemm<T>(Trans, NoTrans, M, N, K, T(1.0),
                               filterData + j * filterOffset,
                               preGrad + j * outputOffset,
                               T(0.0), colData);

                    col2img(out + j * inputOffSize, channel,
                            input_width, input_height,
                            stride[0], stride[1],
                            padding[0],padding[1],
                            kernel[2],kernel[3],
                            dilate[0],dilate[1], colData);

                }
                out += channel * inputShapes->at(DATA)->At(2) * inputShapes->at(DATA)->At(3);
                preGrad += filterNum * inputShapes->at(SELF_OUT)->At(2) * inputShapes->at(SELF_OUT)->At(3);
            }

        } else if (index == 1) {
            const T *inputData = Input<T>(DATA);
            int M = channel / group;
            int K = outputShape->At(2) * outputShape->At(3);
            int N = filterNum / group *  kernel[2] * kernel[3];
            for (int i = 0; i < num; ++i) {
                for (int j = 0; j < group; ++j) {
                    img2col<T>(inputData + j * inputOffSize, channel,
                            input_width, input_height,
                            stride[0], stride[1],
                            padding[0],padding[1],
                            kernel[2],kernel[3],
                            dilate[0],dilate[1], colData);
                    CPUGemm<T>(NoTrans,Trans, M, N, K, T(1.0),
                               preGrad + j * outputOffset,
                               colData,
                               T(0.0),
                               out + j * filterOffset);

                    inputData += channel * input_height * input_width;
                    preGrad += filterNum * inputShapes->at(SELF_OUT)->At(2) * inputShapes->at(SELF_OUT)->At(3);
                }
            }
            Tensor<T> kernel(out, *outputShape);
            Scale(kernel, T(1.0f/num));
        } else if (index == 2) {
            Tensor<T> bias_grad(Output<T>(), *inputShapes->at(BIAS));
            Shape flatten = ShapeN(int(inputShapes->at(PRE_GRAG)->Size()/filterNum), filterNum);
            Tensor<T> pre(Input<T>(PRE_GRAG), flatten);
            SumAdd(pre, 0, bias_grad);
        } else {
            Logger::Global()->Fatal("ConvolutionGradOp do not support other inputs\n");
        }
        return true;
    }


    template<class T, class xpu>
    void ConvolutionGradOp<T, xpu>::AsyncRun() {
        if (xpu::mode == RunMode::kCpu) {
            Run();
        } else {
            if (!RunOnDevice()) {
                Run();
            }
        }
    }

    template<class T, class xpu>
    ConvolutionGradOp<T, xpu>::~ConvolutionGradOp() {

    }

    template<class T, class xpu>
    bool ConvolutionGradOp<T, xpu>::RunOnDevice() {
        return false;
    }



    void ConvolutionOpGradProp::InferShape(std::vector<Shape *> &inShape, Shape *outShape) {
        int index = get<int>(param->args->at("input_idx"));
        switch (index) {
            case 0:
                outShape->reShape(*inShape[2]);
                break;
            case 1:
                outShape->reShape(*inShape[3]);
                break;
            case 2:
                outShape->reShape(*inShape[4]);
                break;
            default:
                Logger::Global()->Fatal("ConvolutionOpGradProp do not support other inputs\n");
                break;
        }

    }
    INIT_OPERATOR_PROPERTY_CREATE(ConvolutionOpGradProp, ConvolutionGradOp, true);


}