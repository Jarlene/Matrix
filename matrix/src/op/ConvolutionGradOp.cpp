//
// Created by Jarlene on 2017/9/8.
//

#include "matrix/include/op/ReduceOp.h"
#include "matrix/include/store/MemoryManager.h"
#include "matrix/include/op/ConvolutionGradOp.h"


namespace matrix {

    template<class T, class Context>
    ConvolutionGradOp<T, Context>::ConvolutionGradOp(Parameter &param) {
        INIT_PARAMS

    }

    template<class T, class Context>
    bool ConvolutionGradOp<T, Context>::Run() {
        ImageOrder order = NCHW;
        if (HasArg("order")) {
            order = GetArgValue<ImageOrder>("order");
        }

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

        int group = 1;
        if (HasArg("group")) {
            group = GetArgValue<int>("group");
        }



        Shape kernel = *inputShapes->at(KERNEL);
        kernel.reShape(ShapeN(filterNum, channel, kernel[0], kernel[1]));
        Shape stride = GetArgValue<Shape>("stride", ShapeN(1, 1));
        Shape padding = GetArgValue<Shape>("padding", ShapeN(0, 0));
        Shape dilate = GetArgValue<Shape>("dilate", ShapeN(1, 1));


        int inputOffSize = channel / group * input_width * input_height;;
        int outputOffset = filterNum / group * outSize;
        int filterOffset = kernel.Size() / group;

        const T *preGrad = Input<T>(PRE_GRAG);
        T *out = Output<T>();
        int index = GetArgValue<int>("input_idx", -1);
        int colSize = channel / group * kernel[2] * kernel[3] * outSize;
        T *colData = static_cast<T *>(MemoryManager::Global()->GetCpuMemoryPool()->dynamicAllocate(colSize * sizeof(T)));

        if (index == 0) {

            const T *filterData = Input<T>(KERNEL);

            int K = filterNum / group;
            int N = outSize;
            int M = channel / group * kernel[2] * kernel[3];

            Value<T>(colSize, colData, T(0.0));
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
            int inputOffSize = filterNum / group * inputShapes->at(PRE_GRAG)->At(2) * inputShapes->at(PRE_GRAG)->At(3);
            int outputOffset = channel / group * outputShape->At(2) * outputShape->At(3);
            int filterOffset = kernel.Size() / group;
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

        } else if (index == 2) {
            Tensor<T> pre(Input<T>(PRE_GRAG), *inputShapes->at(PRE_GRAG));
            Tensor<T> bias_grad(Output<T>(), *inputShapes->at(BIAS));
            Copy<T>(pre, bias_grad);
        } else {
            Logger::Global()->Fatal("ConvolutionGradOp do not support other inputs\n");
        }

        MemoryManager::Global()->GetCpuMemoryPool()->freeMemory(colData, colSize * sizeof(T));
        return true;
    }


    template<class T, class Context>
    void ConvolutionGradOp<T, Context>::AsyncRun() {
        if (Context::mode == RunMode::kCpu) {
            Run();
        } else {
            if (!RunOnDevice()) {
                Run();
            }
        }
    }

    template<class T, class Context>
    ConvolutionGradOp<T, Context>::~ConvolutionGradOp() {

    }

    template<class T, class Context>
    bool ConvolutionGradOp<T, Context>::RunOnDevice() {
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