//
// Created by Jarlene on 2017/8/2.
//

#include "matrix/include/op/ConvolutionOp.h"


namespace matrix {

    template <class T, class Context>
    ConvolutionOp<T, Context>::ConvolutionOp(Parameter &param) {
        INIT_PARAMS
        if (HasArg("col_buffer")) {
            Shape col = GetArgValue<Shape>("col_buffer");
            inputShapes.push_back(&col);
        }
    }

    template <class T, class Context>
    bool ConvolutionOp<T, Context>::Run() {
        if (Inputs().size() <= 2) {
//            Tensor<T> data = Inputs()[DATA]. template GeneratorTensor<T>(inputShapes[DATA]);
//            if (!HasArg("kernel")) {
//                Logger::Global()->Fatal("ConvolutionOp one input must has kernel shape \n");
//            }
//            Shape kShape = GetArgValue<Shape>("kernel");
//            void* kData = MemoryManager::Global()->GetCpuMemoryPool()->dynamicAllocate(kShape.Size() * sizeof(T));
//            Blob weight(kData);
//            input.push_back(weight);
//            Tensor<T>  kernel = weight.template GeneratorTensor<T>(kShape);
//
//            if (HasArg("bias")) {
//                Shape bShape = GetArgValue<Shape>("bias");
//                void* bData = MemoryManager::Global()->GetCpuMemoryPool()->dynamicAllocate(bShape.Size() * sizeof(T));
//                Blob bias(bData);
//                input.push_back(Blob(bias));
//                Tensor<T>  b = bias.template GeneratorTensor<T>(bShape);
//            }
        } else if (Inputs().size() == 3) {

        } else if (Inputs().size() == 4) {
            Tensor<T> data = Inputs()[DATA]-> template GeneratorTensor<T>(inputShapes[DATA]);
            Tensor<T> weight = Inputs()[KERNEL]-> template GeneratorTensor<T>(inputShapes[KERNEL]);
            Tensor<T> bias = Inputs()[BIAS]-> template GeneratorTensor<T>(inputShapes[BIAS]);
            Tensor<T> colBuffer = Inputs()[COLBUFFER]-> template GeneratorTensor<T>(inputShapes[COLBUFFER]);
            Tensor<T> out = Outputs()-> template GeneratorTensor<T>(outputShapes);

            int num = data.GetShape()[0];
            int filterNum = GetArgValue<int>("filter_num");
            ImageOrder  order = GetArgValue<ImageOrder>("order");
            int imageSize = 1;
            int channel = 1;
            if (order == NCHW) {
                channel = data.GetShape()[1];
                imageSize = data.GetShape()[2] * data.GetShape()[3];
            } else {
                imageSize = data.GetShape()[1] * data.GetShape()[2];
                channel = data.GetShape()[3];
            }

            int group = 1;
            if (HasArg("group")) {
                group = GetArgValue<int>("group");
            }

            const int input_offset = channel / group * imageSize;

            const int output_offset = out.Size() / out.GetShape()[0] / group;

            const int filter_offset = inputShapes[KERNEL]->Size() / group;

            for (int i = 0; i < filterNum; ++i) {
                for (int j = 0; j < group; ++j) {



                }
            }

        } else {
            Logger::Global()->Fatal("ConvolutionOp do not support other inputs\n");
        }


        return true;
    }


    template <class T, class Context>
    void ConvolutionOp<T, Context>::AsyncRun() {
        if (Context::mode == RunMode::kCpu) {
            Run();
        } else {
            if (!RunOnDevice()) {
                Run();
            }
        }
    }

    template <class T, class Context>
    ConvolutionOp<T, Context>::~ConvolutionOp() {

    }

    template <class T, class Context>
    bool ConvolutionOp<T, Context>::RunOnDevice() {
        return false;
    }



    template <>
    Operator* CreateOp<CPU>(ConvolutionParam &param, long* size) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new ConvolutionOp<DType, CPU>(param);
            *size = sizeof(DType) * param.outShapes->Size();
        })
        return op;
    }

    template <>
    Operator* CreateOp<GPU>(ConvolutionParam &param, long* size) {
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

    void ConvolutionOpProp::InferShape(std::vector<Shape*> &inShape, Shape *outShape) {

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
            int width  = (in[3] + padding[1] - (dilate[1] * (kernel[1] - 1) + 1)) / stride[1] + 1;
            filter_num = channel;
            if (param->args->count("filter_num")) {
                filter_num = get<int>(param->args->at("filter_num"));
            }
            out.reShape(ShapeN(n, filter_num, height, width));
            int c = channel/group* kernel.Size();

            colBuffer.reShape(ShapeN(c, height, width));
            kernel.reShape(ShapeN(filter_num, channel, kernel_h, kernel_w));
        } else {
            int height = (in[1] + padding[0] - (dilate[0] * (kernel[0] - 1) + 1)) / stride[0] + 1;
            int width  = (in[2] + padding[1] - (dilate[1] * (kernel[1] - 1) + 1)) / stride[1] + 1;
            int channel = in[3];
            filter_num = channel;
            if (param->args->count("filter_num")) {
                filter_num = get<int>(param->args->at("filter_num"));
            }
            out.reShape(ShapeN(n, height, width, filter_num));
            int c = channel/group* kernel.Size();
            colBuffer.reShape(ShapeN(c, height, width));
            kernel.reShape(ShapeN(filter_num, channel, kernel_h, kernel_w));
        }

    }

    Operator *ConvolutionOpProp::CreateOperator(Context context, std::vector<Blob*> &input, Blob* output,
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

    void ConvolutionOpProp::SwitchType(const MatrixType &type) {
        this->param->type = type;
    }
}