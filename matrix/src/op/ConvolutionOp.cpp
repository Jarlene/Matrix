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
        int filterNum = GetArgValue<int>("filter_num");

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

        const int input_offset = channel / group * imageSize;

        const int output_offset = outputShapes->Size() / outputShapes->At(0) / group;

        const int filter_offset = inputShapes[KERNEL]->Size() / group;

        const T *inputData = Input<T>(DATA);
        const T *kernelData = Input<T>(KERNEL);
        T *outputData = Output<T>();

        if (Inputs().size() == 2) {
            Shape colBuffer = *inputShapes[KERNEL + 1];
            T *colData = static_cast<T*>(MemoryManager::Global()->GetCpuMemoryPool()->dynamicAllocate(colBuffer.Size() * sizeof(T)));
            for (int i = 0; i < num; ++i) {
                for (int j = 0; j < group; ++j) {
                    Img2Col<T>(inputData + j * input_offset, *inputShapes[DATA], *inputShapes[KERNEL],
                               GetArgValue<Shape>("stride"),
                               GetArgValue<Shape>("padding"), GetArgValue<Shape>("dilate"), colData, order);
                    int M = filterNum / group;
                    int N = outputShapes->At(2) * outputShapes->At(3);
                    int K = channel / group * inputShapes[KERNEL]->At(2) * inputShapes[KERNEL]->At(3);
                    CPUGemm<T>(NoTrans, NoTrans, M, N, K, T(1.0), kernelData + j * filter_offset, colData,
                               T(0.0), outputData + j * output_offset);
                }
                inputData += channel * imageSize;
                outputData += filterNum * outputShapes->At(2) * outputShapes->At(3);
            }
            MemoryManager::Global()->GetCpuMemoryPool()->freeMemory(colData, colBuffer.Size() * sizeof(T));
        } else if (Inputs().size() == 3) {
            Shape colBuffer = *inputShapes[BIAS + 1];
            T *colData = static_cast<T*>(MemoryManager::Global()->GetCpuMemoryPool()->dynamicAllocate(colBuffer.Size() * sizeof(T)));
            for (int i = 0; i < num; ++i) {
                for (int j = 0; j < group; ++j) {
                    Img2Col<T>(inputData + j * input_offset, *inputShapes[DATA], *inputShapes[KERNEL],
                               GetArgValue<Shape>("stride"),
                               GetArgValue<Shape>("padding"), GetArgValue<Shape>("dilate"), colData, order);
                    int M = filterNum / group;
                    int N = outputShapes->At(2) * outputShapes->At(3);
                    int K = channel / group * inputShapes[KERNEL]->At(2) * inputShapes[KERNEL]->At(3);
                    CPUGemm<T>(NoTrans, NoTrans, M, N, K, T(1.0), kernelData + j * filter_offset, colData,
                               T(0.0), outputData + j * output_offset);
                }
                inputData += channel * imageSize;
                outputData += filterNum * outputShapes->At(2) * outputShapes->At(3);
            }
            MemoryManager::Global()->GetCpuMemoryPool()->freeMemory(colData, colBuffer.Size() * sizeof(T));
        } else if (Inputs().size() == 4) {
            T *colBuff = InputNonConst<T>(COLBUFFER);
            for (int i = 0; i < num; ++i) {
                for (int j = 0; j < group; ++j) {
                    Img2Col<T>(inputData + j * input_offset, *inputShapes[DATA], *inputShapes[KERNEL],
                               GetArgValue<Shape>("stride"),
                               GetArgValue<Shape>("padding"), GetArgValue<Shape>("dilate"), colBuff, order);
                    int M = filterNum / group;
                    int N = outputShapes->At(2) * outputShapes->At(3);
                    int K = channel / group * inputShapes[KERNEL]->At(2) * inputShapes[KERNEL]->At(3);
                    CPUGemm<T>(NoTrans, NoTrans, M, N, K, T(1.0), kernelData + j * filter_offset, colBuff,
                               T(0.0), outputData + j * output_offset);
                }
                inputData += channel * imageSize;
                outputData += filterNum * outputShapes->At(2) * outputShapes->At(3);
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
    bool ConvolutionOp<T, Context>::RunOnDevice() {
        return false;
    }


    template<>
    Operator *CreateOp<CPU>(ConvolutionParam &param, long *size) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new ConvolutionOp<DType, CPU>(param);
            *size = sizeof(DType) * param.outShapes->Size();
        })
        return op;
    }

    template<>
    Operator *CreateOp<GPU>(ConvolutionParam &param, long *size) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new ConvolutionOp<DType, GPU>(param);
            *size = sizeof(DType) * param.outShapes->Size();
        })
        return op;
    }

    ConvolutionOpProp::ConvolutionOpProp() {
        param = new ConvolutionParam(kFloat);
    }

    ConvolutionOpProp::ConvolutionOpProp(const MatrixType &type) {
        param = new ConvolutionParam(type);
    }

    ConvolutionOpProp::~ConvolutionOpProp() {
        delete param;
    }

    void ConvolutionOpProp::InferShape(std::vector<Shape *> &inShape, Shape *outShape) {

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
        Shape out = *outShape;
        Shape colBuffer;
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
        inShape.push_back(&colBuffer);
    }

    Operator *ConvolutionOpProp::CreateOperator(Context context, std::vector<Blob *> &input, Blob *output,
                                                std::vector<Shape *> &inShape, Shape *outShape,
                                                std::map<std::string, Any> &args) {
        param->args = &args;
        param->inputs = input;
        param->outputs = output;
        InferShape(inShape, outShape);
        param->inputShapes = inShape;
        param->outShapes = outShape;
        BIND_DISPATCH(CreateOp, *param, &memorySize);
    }

    void ConvolutionOpProp::SwitchType(const MatrixType &type) {
        this->param->type = type;
    }
}