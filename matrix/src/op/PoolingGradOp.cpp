//
// Created by Jarlene on 2017/11/6.
//


#include "matrix/include/store/MemoryManager.h"
#include "matrix/include/op/PoolingGradOp.h"
#include "matrix/include/op/ReduceOp.h"

namespace matrix {

    template <class T, class Context>
    PoolingGradOp<T, Context>::PoolingGradOp(Parameter &param) {
        INIT_PARAMS
    }

    template <class T, class Context>
    bool PoolingGradOp<T, Context>::Run() {
        PoolType  type = GetArgValue<PoolType>("type", kMax);
        auto filter = GetArgValue<Shape>("filter");
        auto stride = GetArgValue<Shape>("stride", ShapeN(1, 1));
        auto padding = GetArgValue<Shape>("padding", ShapeN(0 ,0));
        auto dilate = GetArgValue<Shape>("dilate", ShapeN(1, 1));

        int batch_size = inputShapes[INPUT]->At(0);
        int channel = inputShapes[INPUT]->At(1);
        int input_width = inputShapes[INPUT]->At(2);
        int input_height = inputShapes[INPUT]->At(3);
        int imageSize = input_height * input_width;

        int group = 1;
        if (HasArg("group")) {
            group = GetArgValue<int>("group");
        }

        const int input_offset = channel / group * imageSize;
        const T * pre_grad = Input<T>(PRE_GRAG);
        const T * input = Input<T>(INPUT);
        T * out = Output<T>();


        switch (type) {
            case kMax:
            {
                if (!HasArg("max_index")) {
                    Logger::Global()->Fatal("PoolingGradOp--> not max_index in param, please check\n");
                }
                Tensor<int> maxIndex = GetArgValue<Tensor<int>>("max_index");
            }
                break;
            case kAvg:
            {
                for (int i = 0; i < batch_size; ++i) {
                    for (int j = 0; j < group; ++j) {

                    }
                }
            }
                break;
            default:
                Logger::Global()->Fatal("PoolingOp do not support PoolType\n");
                break;
        }
        return true;
    }

    template <class T, class Context>
    void PoolingGradOp<T, Context>::AsyncRun() {
        if (Context::mode == RunMode::kCpu) {
            Run();
        } else {
            if (!RunOnDevice()) {
                Run();
            }
        }
    }

    template <class T, class Context>
    bool PoolingGradOp<T, Context>::RunOnDevice() {
        return false;
    }

    template <class T, class Context>
    PoolingGradOp<T, Context>::~PoolingGradOp() {

    }



    template <>
    Operator* CreateOp<CPU>(PoolingGradParam &param, long *size) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new PoolingGradOp<DType, CPU>(param);
            *size = sizeof(DType) * param.outShapes->Size();
        })
        return op;
    }

    template <>
    Operator* CreateOp<GPU>(PoolingGradParam &param, long *size) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new PoolingGradOp<DType, GPU>(param);
            *size = sizeof(DType) * param.outShapes->Size();
        })
        return op;
    }



    PoolingGradOpProp::PoolingGradOpProp() {
        param = new PoolingGradParam(kFloat);
    }

    PoolingGradOpProp::PoolingGradOpProp(const MatrixType &type) {
        param = new PoolingGradParam(type);
    }

    PoolingGradOpProp::~PoolingGradOpProp() {
        delete param;
    }

    void PoolingGradOpProp::InferShape(std::vector<Shape*> &inShape, Shape* outShape) {
        Shape padding = ShapeN(0, 0);
        Shape stride = ShapeN(1, 1);
        Shape dilate = ShapeN(1, 1);
        Shape filter = ShapeN(1, 1);
        if (param->args->count("filter")) {
            auto s = get<Shape>(param->args->at("filter"));
            filter.reShape(s);
            s.reShape(ShapeN(inShape[2]->At(0), inShape[2]->At(1), filter[0], filter[1]));
            filter.reShape(get<Shape>(param->args->at("filter")));
        } else {
            Logger::Global()->Fatal("PoolingOp cant not support no filter \n");
        }
        if (param->args->count("stride")) {
            stride.reShape(get<Shape>(param->args->at("stride")));
        } else {
            param->args->insert(std::pair<std::string, Any>("stride", stride));
        }
        if (param->args->count("padding")) {
            padding.reShape(get<Shape>(param->args->at("padding")));
        } else {
            param->args->insert(std::pair<std::string, Any>("padding", padding));
        }
        if (param->args->count("dilate")) {
            dilate.reShape(get<Shape>(param->args->at("dilate")));
        } else {
            param->args->insert(std::pair<std::string, Any>("dilate", dilate));
        }
        int w = (inShape[0]->At(2) + 2 * padding[0] - (dilate[0] * (filter[0] - 1) + 1) )/stride[0] + 1 ;
        int h = (inShape[0]->At(3) + 2 * padding[1] - (dilate[1] * (filter[1] - 1) + 1) )/stride[1] + 1 ;
        outShape->reShape(ShapeN(inShape[0]->At(0), inShape[0]->At(1), w, h));
    }

    Operator *PoolingGradOpProp::CreateOperator(Context context, std::vector<Blob*> &input, Blob* output,
                                            std::vector<Shape*> &inShape, Shape* outShape,
                                            std::map<std::string, Any> &args) {
        param->args = &args;
        param->inputs = input;
        param->outputs = output;
        InferShape(inShape, outShape);
        param->inputShapes = inShape;
        param->outShapes = outShape;
        BIND_DISPATCH(CreateOp, *param, &memorySize);
    }

    void PoolingGradOpProp::SwitchType(const MatrixType &type) {
        this->param->type = type;
    }

}