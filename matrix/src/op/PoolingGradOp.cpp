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

        int batch_size = inputShapes->at(INPUT)->At(0);
        int channel = inputShapes->at(INPUT)->At(1);
        int input_width = inputShapes->at(INPUT)->At(2);
        int input_height = inputShapes->at(INPUT)->At(3);
        int imageSize = input_height * input_width;

        int output_width = outputShape->At(2);
        int output_height = outputShape->At(3);

        const T * pre_grad = Input<T>(PRE_GRAG);
        T * out = Output<T>();
        Value<T>(outputShape->Size(), out, T(0));
        switch (type) {
            case kMax:
            {
                if (!HasArg("max_index")) {
                    Logger::Global()->Fatal("PoolingGradOp--> not max_index in param, please check\n");
                }
                Tensor<int> maxIndex = GetArgValue<Tensor<int>>("max_index");
                for (int i = 0; i < batch_size; ++i) {
                    for (int j = 0; j < channel; ++j) {
                        for (int k = 0; k < inputShapes->at(PRE_GRAG)->Size(); ++k) {
                            const int *idx = maxIndex.Data(k);
                            out[*idx] += pre_grad[k];
                        }
                    }
                    pre_grad +=  output_width * output_height;
                    out +=  imageSize;
                }
            }
                break;
            case kAvg:
            {
                for (int i = 0; i < batch_size; ++i) {
                    for (int j = 0; j < channel; ++j) {
                        for (int ph = 0; ph < output_height; ++ph) {
                            int hstart = std::max(ph * stride[1] - padding[1], 0);
                            int hend = std::min(hstart + filter[3], input_height);
                            for (int pw = 0; pw < output_width; ++pw) {
                                int wstart = std::max(pw * stride[0] - padding[0], 0);
                                int wend = std::min(wstart + filter[2], input_width);
                                int pool_size = (hend - hstart) * (wend - wstart);
                                T scale = 1.0 / pool_size;
                                for (int h = hstart; h < hend; ++h) {
                                    for (int w = wstart; w < wend; ++w) {
                                        out[h * input_width + w] += (pre_grad[ph * output_width + pw] * scale);
                                    }
                                }
                            }
                        }
                    }
                    pre_grad += output_width * output_height;
                    out +=  imageSize;
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



    void PoolingGradOpProp::InferShape(std::vector<Shape*> &inShape, Shape* outShape) {
        Shape padding = ShapeN(0, 0);
        Shape stride = ShapeN(1, 1);
        Shape dilate = ShapeN(1, 1);
        Shape filter = ShapeN(1, 1);
        if (param->args->count("filter")) {
            auto s = get<Shape>(param->args->at("filter"));
            filter.reShape(s);
            get<Shape>(param->args->at("filter")).reShape(ShapeN(inShape[2]->At(0), inShape[2]->At(1), filter[0], filter[1]));
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
        outShape->reShape(*inShape[2]);
    }

    INIT_OPERATOR_PROPERTY_CREATE(PoolingGradOpProp, PoolingGradOp, true);


}