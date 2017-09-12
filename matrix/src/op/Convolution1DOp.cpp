//
// Created by Jarlene on 2017/9/6.
//

#include "matrix/include/op/Convolution1DOp.h"


namespace matrix {

    template<class T, class Context>
    Convolution1DOp<T, Context>::Convolution1DOp(Parameter &param) {
        INIT_PARAMS
    }

    template<class T, class Context>
    bool Convolution1DOp<T, Context>::Run() {
        int num = inputShapes[DATA]->At(0);
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
            channel = inputShapes[DATA]->At(1);
        } else {
            channel = inputShapes[DATA]->At(3);
        }

        Shape kernel = *inputShapes[KERNEL];
        Shape stride = GetArgValue<Shape>("stride");
        Shape padding = GetArgValue<Shape>("padding");
        Shape dilate = GetArgValue<Shape>("dilate");

        NaiveConv<T>(inputData, num, channel,
                     inputShapes[DATA]->At(2), inputShapes[DATA]->At(3),
                     stride[0], stride[1],
                     padding[0], padding[1],
                     kernel[2], kernel[3],
                     dilate[0], dilate[1],
                     filterNum, kernelData, outputData);

        if (Inputs().size() == 3) {
            Tensor<T> out = Outputs()->template GeneratorTensor<T>(outputShapes);
            Tensor<T> bias = Inputs()[BIAS]->template GeneratorTensor<T>(inputShapes[BIAS]);
            Add<T>(out, bias, out);
        } else {
            Logger::Global()->Fatal("ConvolutionOp do not support other inputs\n");
        }
        return true;
    }


    template<class T, class Context>
    void Convolution1DOp<T, Context>::AsyncRun() {
        if (Context::mode == RunMode::kCpu) {
            Run();
        } else {
            if (!RunOnDevice()) {
                Run();
            }
        }
    }

    template<class T, class Context>
    Convolution1DOp<T, Context>::~Convolution1DOp() {

    }

    template<class T, class Context>
    bool Convolution1DOp<T, Context>::RunOnDevice() {
        return false;
    }


    template<>
    Operator *CreateOp<CPU>(Convolution1DParam &param, long *size) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new Convolution1DOp<DType, CPU>(param);
            *size = sizeof(DType) * param.outShapes->Size();
        })
        return op;
    }

    template<>
    Operator *CreateOp<GPU>(Convolution1DParam &param, long *size) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new Convolution1DOp<DType, GPU>(param);
            *size = sizeof(DType) * param.outShapes->Size();
        })
        return op;
    }

    Convolution1DOpProp::Convolution1DOpProp() {
        param = new Convolution1DParam(kFloat);
    }

    Convolution1DOpProp::Convolution1DOpProp(const MatrixType &type) {
        param = new Convolution1DParam(type);
    }

    Convolution1DOpProp::~Convolution1DOpProp() {
        delete param;
    }

    void Convolution1DOpProp::InferShape(std::vector<Shape *> &inShape, Shape *outShape) {

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
            int height = (in[2] + 2 * padding[0] - (dilate[0] * (kernel[0] - 1) + 1)) / stride[0] + 1;
            int width = (in[3] + 2 * padding[1] - (dilate[1] * (kernel[1] - 1) + 1)) / stride[1] + 1;
            filter_num = channel;
            if (param->args->count("filter_num")) {
                filter_num = get<int>(param->args->at("filter_num"));
            }
            outShape->reShape(ShapeN(n, filter_num, height, width));
            int c = channel / group * kernel.Size();

            colBuffer.reShape(ShapeN(c, height, width));
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
            int c = channel / group * kernel.Size();
            colBuffer.reShape(ShapeN(c, height, width));
            inShape[1]->reShape(ShapeN(filter_num, channel, kernel_h, kernel_w));
        }
        inShape.push_back(&colBuffer);

    }

    Operator *Convolution1DOpProp::CreateOperator(Context context, std::vector<Blob *> &input, Blob *output,
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

    void Convolution1DOpProp::SwitchType(const MatrixType &type) {
        this->param->type = type;
    }
}