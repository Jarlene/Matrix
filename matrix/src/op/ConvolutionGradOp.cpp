//
// Created by Jarlene on 2017/9/8.
//

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

        int num = inputShapes[DATA]->At(0);

        int imageSize = 1;
        int channel = 1;
        if (order == NCHW) {
            channel = inputShapes[DATA]->At(1);
            imageSize = inputShapes[DATA]->At(2) * inputShapes[DATA]->At(3);
        } else {
            imageSize = inputShapes[DATA]->At(1) * inputShapes[DATA]->At(2);
            channel = inputShapes[DATA]->At(3);
        }

        int filterNum = GetArgValue<int>("filter_num", channel);

        int group = 1;
        if (HasArg("group")) {
            group = GetArgValue<int>("group");
        }



        Shape kernel = *inputShapes[KERNEL];
        Shape stride = GetArgValue<Shape>("stride", ShapeN(1, 1));
        Shape padding = GetArgValue<Shape>("padding", ShapeN(0, 0));
        Shape dilate = GetArgValue<Shape>("dilate", ShapeN(1, 1));


        int inputOffSize = channel / group * inputShapes[DATA]->At(2) * inputShapes[DATA]->At(3);
        int outputOffset = filterNum / group * inputShapes[SELF_OUT]->At(2) * inputShapes[SELF_OUT]->At(3);
        int filterOffset = kernel.Size() / group;

        const T *preGrad = Inputs()[PRE_GRAG]-> template Get<T>();
        T *out = Output<T>();
        int index = GetArgValue<int>("input_idx", -1);
        if (index == 0) {

            const T *filterData = Inputs()[KERNEL]-> template Get<T>();

            int K = filterNum / group;
            int N = inputShapes[SELF_OUT]->At(2) * inputShapes[SELF_OUT]->At(3);
            int M = channel / group * kernel[2] * kernel[3];

            int colSize = channel / group * kernel[2] * kernel[3] * inputShapes[SELF_OUT]->At(2) * inputShapes[SELF_OUT]->At(3);
            T *colData = static_cast<T *>(MemoryManager::Global()->GetCpuMemoryPool()->dynamicAllocate(
                    colSize * sizeof(T)));
            Value<T>(colSize, colData, T(0.0));
            for (int i = 0; i < num; ++i) {
                for (int j = 0; j < group; ++j) {
                    CPUGemm<T>(Trans, NoTrans, M, N, K, T(1.0),
                               filterData + j * filterOffset,
                               preGrad + j * outputOffset,
                               T(0.0), colData);

                    col2img(out + j * inputOffSize, filterNum,
                            outputShapes[PRE_GRAG].At(2), outputShapes[PRE_GRAG].At(3),
                            stride[0], stride[1],
                            padding[0],padding[1],
                            kernel[2],kernel[3],
                            dilate[0],dilate[1], colData);

                }
                out += channel * inputShapes[DATA]->At(2) * inputShapes[DATA]->At(3);
                preGrad += filterNum * inputShapes[SELF_OUT]->At(2) * inputShapes[SELF_OUT]->At(3);
            }
            MemoryManager::Global()->GetCpuMemoryPool()->freeMemory(colData, colSize * sizeof(T));
        } else if (index == 1) {
            int inputOffSize = filterNum / group * inputShapes[PRE_GRAG]->At(2) * inputShapes[PRE_GRAG]->At(3);
            int outputOffset = channel / group * outputShapes->At(2) * outputShapes->At(3);
            int filterOffset = kernel.Size() / group;
            const T *inputData = Inputs()[DATA]-> template Get<T>();
            int M = channel / group;
            int K = outputShapes->At(2) * outputShapes->At(3);
            int N = filterNum / group *  kernel[2] * kernel[3];
            for (int i = 0; i < num; ++i) {
                for (int j = 0; j < group; ++j) {
//                    img2col(inputData + j * inputOffSize, filterNum, );
                }
            }

        } else if (index == 2) {

        } else {
            Logger::Global()->Fatal("ConvolutionGradOp do not support other inputs\n");
        }


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


    template<>
    Operator *CreateOp<CPU>(ConvolutionGradParam &param, long *size) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new ConvolutionGradOp<DType, CPU>(param);
            *size = sizeof(DType) * param.outShapes->Size();
        })
        return op;
    }

    template<>
    Operator *CreateOp<GPU>(ConvolutionGradParam &param, long *size) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new ConvolutionGradOp<DType, GPU>(param);
            *size = sizeof(DType) * param.outShapes->Size();
        })
        return op;
    }

    ConvolutionOpGradProp::ConvolutionOpGradProp() {
        param = new ConvolutionGradParam(kFloat);
    }

    ConvolutionOpGradProp::ConvolutionOpGradProp(const MatrixType &type) {
        param = new ConvolutionGradParam(type);
    }

    ConvolutionOpGradProp::~ConvolutionOpGradProp() {
        delete param;
    }

    void ConvolutionOpGradProp::InferShape(std::vector<Shape *> &inShape, Shape *outShape) {
        int index = get<int>(param->args->at("input_idx"));
        switch (index) {
            case 0:
                outShape->reShape(*inShape[1]);
                break;
            case 1:
                outShape->reShape(*inShape[2]);
                break;
            case 2:
                outShape->reShape(*inShape[3]);
                break;
            default:
                Logger::Global()->Fatal("ConvolutionOpGradProp do not support other inputs\n");
                break;
        }

    }

    Operator *ConvolutionOpGradProp::CreateOperator(Context context, std::vector<Blob*> &input, Blob* output,
                                                    std::vector<Shape*> &inShape, Shape *outShape,
                                                    std::map<std::string, Any> &args) {
        param->args = &args;
        param->inputs = input;
        param->outputs = output;
        InferShape(inShape, outShape);
        param->inputShapes = inShape;
        param->outShapes = outShape;
        BIND_DISPATCH(CreateOp, *param, &memorySize);
    }

    void ConvolutionOpGradProp::SwitchType(const MatrixType &type) {
        this->param->type = type;
    }
}