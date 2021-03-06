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
        int imageSize = 1;
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
        imageSize = input_width * input_height;
        int filterNum = GetArgValue<int>("filter_num", channel);
        int group = GetArgValue<int>("group", 1);

        Shape kernel = *inputShapes->at(KERNEL);
        Shape stride = GetArgValue<Shape>("stride", ShapeN(1, 1));
        Shape padding = GetArgValue<Shape>("padding", ShapeN(0, 0));
        Shape dilate = GetArgValue<Shape>("dilate", ShapeN(1, 1));


        int inputOffSize = channel / group * imageSize;
        int outputOffset = filterNum / group * outSize;
        int filterOffset = kernel.Size() / group;

        T *preGrad = Input<T>(PRE_GRAG);
        const T *self_out = Input<T>(SELF_OUT);
        T *out = Output<T>();
        int index = GetArgValue<int>("input_idx", -1);
        Value(outputShape->Size(), out, T(0));
        T *colData = Input<T>(InputSize() - 1);
        if (HasArg("activation_type")) {
            auto actType = GetArgValue<ActType>("activation_type");
            switch (actType) {
                case kSigmoid:
                    SigmoidGrad<T>(outputShape->Size(), self_out, preGrad, preGrad);
                    break;
                case kTanh:
                    TanhGrad<T>(outputShape->Size(), self_out, preGrad, preGrad);
                    break;
                case kRelu:
                    ReluGrad<T>(outputShape->Size(), self_out, preGrad, preGrad);
                    break;
                default:
                    Logger::Global()->Fatal("ConvolutionGradOp activation_type not support \n");
                    break;
            }
        }
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

                    Col2Img<T, 0>(colData, channel,
                                  input_width, input_height,
                                  kernel[2],kernel[3],
                                  dilate[0],dilate[1],
                                  padding[0], padding[1], padding[0], padding[1],
                                  stride[0], stride[1], out + j * inputOffSize);

                }
                out += inputOffSize * group;
                preGrad += outputOffset * group;
            }

        } else if (index == 1) {
            const T *inputData = Input<T>(DATA);
            int M = filterNum / group;
            int K = outSize;
            int N = channel / group *  kernel[2] * kernel[3];
            for (int i = 0; i < num; ++i) {
                for (int j = 0; j < group; ++j) {
                    Img2Col<T, 0>(inputData + j * inputOffSize, channel,
                                  input_width, input_height,
                                  kernel[2],kernel[3],
                                  dilate[0],dilate[1],
                                  padding[0], padding[1], padding[0], padding[1],
                                  stride[0], stride[1], colData);
                    CPUGemm<T>(NoTrans,Trans, M, N, K, T(1.0),
                               preGrad + j * outputOffset,
                               colData,
                               T(1.0),
                               out + j * filterOffset);

                }
                inputData += inputOffSize * group;
                preGrad += outputOffset * group;
            }
        } else if (index == 2) {
            for (int i = 0; i < num; ++i) {
                for (int j = 0; j < filterNum; ++j) {
                    for (int k = 0; k < outSize; ++k) {
                        out[j] += preGrad[i * filterNum * outSize + j * outSize + k];
                    }
                }
            }
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

    template<class T, class xpu>
    bool ConvolutionGradOp<T, xpu>::ShareNodes(std::function<void(std::initializer_list<Shape *> shapes)> func) {
        int index = GetArgValue<int>("input_idx", -1);
        if (index == 0 || index == 2) {
            return false;
        }
        bool with_bias = GetArgValue<bool>("with_bias", false);
        if (InputSize() == 6) {
            with_bias = true;
        } else if (InputSize() == 5) {
            with_bias = false;
        }
        if ((InputSize() < 7 && with_bias) || (InputSize() < 6 && !with_bias)) {
            int group = GetArgValue<int>("group", 1);
            int channel = inputShapes->at(KERNEL)->At(1);
            auto colShape = ShapeN(channel/group, inputShapes->at(KERNEL)->At(2) , inputShapes->at(KERNEL)->At(3),
                                   inputShapes->at(SELF_OUT)->At(2) , inputShapes->at(SELF_OUT)->At(3));
            func({&colShape});
            return true;
        }
        return false;
    }


    void ConvolutionOpGradProp::InferShape(std::vector<Shape *> &inShape, Shape *outShape) {
        int index = param->GetArgValue<int>("input_idx");
        outShape->reShape(*inShape[index + 2]);
    }
    INIT_OPERATOR_PROPERTY_CREATE(ConvolutionOpGradProp, ConvolutionGradOp, true);


}