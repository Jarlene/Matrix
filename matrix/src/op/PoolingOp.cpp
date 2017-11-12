//
// Created by Jarlene on 2017/8/2.
//

#include "matrix/include/store/MemoryManager.h"
#include "matrix/include/op/PoolingOp.h"
#include "matrix/include/op/ReduceOp.h"

namespace matrix {

    template <class T, class Context>
    PoolingOp<T, Context>::PoolingOp(Parameter &param) {
        INIT_PARAMS
    }

    template <class T, class Context>
    bool PoolingOp<T, Context>::Run() {
        PoolType  type = GetArgValue<PoolType>("type", kMax);


        auto filter = GetArgValue<Shape>("filter");
        auto stride = GetArgValue<Shape>("stride", ShapeN(1, 1));
        auto padding = GetArgValue<Shape>("padding", ShapeN(0 ,0));
        auto dilate = GetArgValue<Shape>("dilate", ShapeN(1, 1));

        int batch_size = inputShapes[0]->At(0);
        int channel = inputShapes[0]->At(1);
        int input_width = inputShapes[0]->At(2);
        int input_height = inputShapes[0]->At(3);


        int group = 1;
        if (HasArg("group")) {
            group = GetArgValue<int>("group");
        }

        const T* input = Input<T>(0);
        T* out = Output<T>();


        if (type == kMax) {
            int *maxIndex = static_cast<int*>(MemoryManager::Global()->GetCpuMemoryPool()->dynamicAllocate(
                    batch_size * outputShapes->At(2) * outputShapes->At(3) * sizeof(int)));
            Shape maxShape;
            maxShape.Append(batch_size * outputShapes->At(2) * outputShapes->At(3));
            Tensor<int> maxTensor(maxIndex, maxShape);
            pooling2D(input, batch_size, channel/group, input_width, input_height, stride[0], stride[1],
                      padding[0], padding[1],filter[2], filter[3], dilate[0], dilate[1],out,type, maxIndex);
            args->insert(std::pair<std::string, Any>("max_index", maxTensor));
        } else {
            pooling2D(input, batch_size, channel/group, input_width, input_height, stride[0], stride[1],
                      padding[0], padding[1],filter[2], filter[3], dilate[0], dilate[1], out, type);
        }
        return true;
    }

    template <class T, class Context>
    void PoolingOp<T, Context>::AsyncRun() {
        if (Context::mode == RunMode::kCpu) {
            Run();
        } else {
            if (!RunOnDevice()) {
                Run();
            }
        }
    }

    template <class T, class Context>
    bool PoolingOp<T, Context>::RunOnDevice() {
        return false;
    }

    template <class T, class Context>
    PoolingOp<T, Context>::~PoolingOp() {

    }



    template <>
    Operator* CreateOp<CPU>(Parameter &param, long *size) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new PoolingOp<DType, CPU>(param);
            *size = sizeof(DType) * param.outShapes->Size();
        })
        return op;
    }

    template <>
    Operator* CreateOp<GPU>(Parameter &param, long *size) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new PoolingOp<DType, GPU>(param);
            *size = sizeof(DType) * param.outShapes->Size();
        })
        return op;
    }



    PoolingOpProp::PoolingOpProp() {
        param = new Parameter(kFloat);
    }

    PoolingOpProp::PoolingOpProp(const MatrixType &type) {
        param = new Parameter(type);
    }

    PoolingOpProp::~PoolingOpProp() {
        delete param;
    }

    void PoolingOpProp::InferShape(std::vector<Shape*> &inShape, Shape* outShape) {
        Shape padding = ShapeN(0, 0);
        Shape stride = ShapeN(1, 1);
        Shape dilate = ShapeN(1, 1);
        Shape filter = ShapeN(1, 1);
        if (param->args->count("filter")) {
            auto s = get<Shape>(param->args->at("filter"));
            filter.reShape(s);
            get<Shape>(param->args->at("filter")).reShape(ShapeN(inShape[0]->At(0), inShape[0]->At(1), filter[0], filter[1]));
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

    Operator *PoolingOpProp::CreateOperator(Context context, std::vector<Blob*> &input, Blob* output,
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


}