//
// Created by Jarlene on 2017/8/2.
//

#include "matrix/include/store/MemoryManager.h"
#include "matrix/include/op/ConvolutionOp.h"


namespace matrix {

    template<class T, class Context>
    ConvolutionOp<T, Context>::ConvolutionOp(Parameter &param) {
        INIT_PARAMS

    }

    template<class T, class Context>
    bool ConvolutionOp<T, Context>::Run() {

        int num = inputShapes[DATA]->At(0);

        ImageOrder order = NCHW;
        if (HasArg("order")) {
            order = GetArgValue<ImageOrder>("order");
        }
        int imageSize = 1;
        int channel = 1;
        if (order == NCHW) {
            channel = inputShapes[DATA]->At(1);
            imageSize = inputShapes[DATA]->At(2) * inputShapes[DATA]->At(3);
        } else {
            imageSize = inputShapes[DATA]->At(1) * inputShapes[DATA]->At(2);
            channel = inputShapes[DATA]->At(3);
        }

        int group = 1;
        if (HasArg("group")) {
            group = GetArgValue<int>("group");
        }
        int filterNum = GetArgValue<int>("filter_num", channel);

        Shape kernel;
        Shape stride = GetArgValue<Shape>("stride", ShapeN(1, 1));
        Shape padding = GetArgValue<Shape>("padding", ShapeN(0, 0));
        Shape dilate = GetArgValue<Shape>("dilate", ShapeN(1, 1));


        const int input_offset = channel / group * imageSize;

        const int output_offset = outputShape->Size() / outputShape->At(0) / group;

        const int filter_offset = kernel.Size() / group;

        const T *inputData = Input<T>(DATA);
        T *outputData = Output<T>();
        if (input.size() == 2) {
            const T *kernelData = Input<T>(KERNEL);
            kernel = *inputShapes[KERNEL];
            int colSize = channel / group * kernel.Size() * outputShape->At(2) * outputShape->At(3);
            T *colData = static_cast<T *>(MemoryManager::Global()->GetCpuMemoryPool()->dynamicAllocate(
                    colSize * sizeof(T)));
            for (int i = 0; i < num; ++i) {
                for (int j = 0; j < group; ++j) {
                    Img2Col<T>(inputData + j * input_offset, *inputShapes[DATA],
                               kernel, stride,
                               padding, dilate,
                               colData, order);
                    int M = filterNum / group;
                    int N = outputShape->At(2) * outputShape->At(3);
                    int K = channel / group * kernel.At(2) * kernel.At(3);
                    CPUGemm<T>(NoTrans, NoTrans, M, N, K, T(1.0), kernelData + j * filter_offset, colData,
                               T(0.0), outputData + j * output_offset);
                }
                inputData += channel * imageSize;
                outputData += filterNum * outputShape->At(2) * outputShape->At(3);
            }
            MemoryManager::Global()->GetCpuMemoryPool()->freeMemory(colData, colSize * sizeof(T));
        } else if (input.size() == 3) {
            const T *kernelData = Input<T>(KERNEL);
            kernel = *inputShapes[KERNEL];
            int colSize = channel / group * kernel.Size() * outputShape->At(2) * outputShape->At(3);
            T *colData = static_cast<T *>(MemoryManager::Global()->GetCpuMemoryPool()->dynamicAllocate(
                    colSize * sizeof(T)));
            for (int i = 0; i < num; ++i) {
                for (int j = 0; j < group; ++j) {
                    Img2Col<T>(inputData + j * input_offset, *inputShapes[DATA],
                               kernel, stride,
                               padding, dilate,
                               colData, order);
                    int M = filterNum / group;
                    int N = outputShape->At(2) * outputShape->At(3);
                    int K = channel / group * kernel.At(2) * kernel.At(3);
                    CPUGemm<T>(NoTrans, NoTrans, M, N, K, T(1.0), kernelData + j * filter_offset, colData,
                               T(0.0), outputData + j * output_offset);
                }
                inputData += channel * imageSize;
                outputData += filterNum * outputShape->At(2) * outputShape->At(3);
            }

            Tensor<T> out(Output<T>(), *outputShape);
            Tensor<T> bias(Input<T>(BIAS), *inputShapes[BIAS]);
            Add<T>(out, bias, out);
            MemoryManager::Global()->GetCpuMemoryPool()->freeMemory(colData, colSize * sizeof(T));
        } else if (input.size() == 4) {
            const T *kernelData = Input<T>(KERNEL);
            kernel = *inputShapes[KERNEL];
            T *colBuff = InputNonConst<T>(COLBUFFER);
            for (int i = 0; i < num; ++i) {
                for (int j = 0; j < group; ++j) {
                    Img2Col<T>(inputData + j * input_offset, *inputShapes[DATA],
                               kernel, stride,
                               padding, dilate,
                               colBuff, order);
                    int M = filterNum / group;
                    int N = outputShape->At(2) * outputShape->At(3);
                    int K = channel / group * kernel.At(2) * kernel.At(3);
                    CPUGemm<T>(NoTrans, NoTrans, M, N, K, T(1.0), kernelData + j * filter_offset, colBuff,
                               T(0.0), outputData + j * output_offset);
                }
                inputData += channel * imageSize;
                outputData += filterNum * outputShape->At(2) * outputShape->At(3);
            }

        } else {
            Logger::Global()->Fatal("ConvolutionOp do not support other inputs\n");
        }
        return true;
    }


    template<class T, class Context>
    void ConvolutionOp<T, Context>::AsyncRun() {
        if (Context::mode == RunMode::kCpu) {
            Run();
        } else {
            if (!RunOnDevice()) {
                Run();
            }
        }
    }

    template<class T, class Context>
    ConvolutionOp<T, Context>::~ConvolutionOp() {

    }

    template<class T, class Context>
    void ConvolutionOp<T, Context>::VariableNode(std::function<void(std::initializer_list<Shape *> shapes)> func) {
        if (inputShapes.size() == 1) {
            if (!HasArg("filter")) {
                Logger::Global()->Fatal("ConvolutionOp no filter for input");
            }
            Shape filter = GetArgValue<Shape>("filter");
            if (HasArg("with_bias")) {
                Shape bias;
                bias.Append(inputShapes[0]->At(0));
                func({&filter, &bias});
            } else {
                func({&filter});
            }
        }
    };

    template<class T, class Context>
    bool ConvolutionOp<T, Context>::RunOnDevice() {
        return false;
    }


    ConvolutionOpProp::ConvolutionOpProp() {
        param = new Parameter(kFloat);
    }

    ConvolutionOpProp::ConvolutionOpProp(const MatrixType &type) {
        param = new Parameter(type);
    }

    ConvolutionOpProp::~ConvolutionOpProp() {
        delete param;
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


        int group = 1;
        if (param->args->count("group")) {
            group = get<int>(param->args->at("group"));
        }
        int filter_num = 1;


        Shape in = *inShape[0];
        Shape kernel = *inShape[1];

        int n = in[0];
        int kernel_h = kernel[0];
        int kernel_w = kernel[1];
        if (order == NCHW) {
            int channel = in[1];
            int height = (in[2] + 2 * padding[0] - (dilate[0] * (kernel[0] - 1) + 1)) / stride[0] + 1;
            int width = (in[3] + 2 * padding[1] - (dilate[1] * (kernel[1] - 1) + 1)) / stride[1] + 1;
            filter_num = channel;
            if (param->args->count("filter_num")) {
                filter_num = get<int>(param->args->at("filter_num"));
            }
            outShape->reShape(ShapeN(n, filter_num, height, width));
            inShape[1]->reShape(ShapeN(filter_num, channel, kernel_h, kernel_w));
        } else {
            int height = (in[1] + 2 * padding[0] - (dilate[0] * (kernel[0] - 1) + 1)) / stride[0] + 1;
            int width = (in[2] + 2 * padding[1] - (dilate[1] * (kernel[1] - 1) + 1)) / stride[1] + 1;
            int channel = in[3];
            filter_num = channel;
            if (param->args->count("filter_num")) {
                filter_num = get<int>(param->args->at("filter_num"));
            }
            outShape->reShape(ShapeN(n, height, width, filter_num));
            inShape[1]->reShape(ShapeN(filter_num, channel, kernel_h, kernel_w));
        }
    }

    Operator *ConvolutionOpProp::CreateOperator(Context context, std::vector<void *> &input, void *output,
                                                std::vector<Shape *> &inShape, Shape *outShape,
                                                std::map<std::string, Any> &args) {
        param->args = &args;
        param->inputs = input;
        param->output = output;
        InferShape(inShape, outShape);
        param->inputShapes = inShape;
        param->outShape = outShape;
        CREATE_OPERATOR(param, ConvolutionOp, {
            memorySize = sizeof(DType) * param->outShape->Size();
        })
    }


}