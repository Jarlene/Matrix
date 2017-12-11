//
// Created by Jarlene on 2017/11/6.
//


#include "matrix/include/store/MemoryManager.h"
#include "matrix/include/op/PoolingGradOp.h"
#include "matrix/include/op/ReduceOp.h"

namespace matrix {

    template <class T, class xpu>
    PoolingGradOp<T, xpu>::PoolingGradOp(Parameter &param) {
        INIT_PARAMS
    }

    template <class T, class xpu>
    bool PoolingGradOp<T, xpu>::Run() {
        PoolType  type = GetArgValue<PoolType>("type", kMax);
        auto filter = GetArgValue<Shape>("filter");
        auto stride = GetArgValue<Shape>("stride", ShapeN(1, 1));
        auto padding = GetArgValue<Shape>("padding", ShapeN(0 ,0));
        auto dilate = GetArgValue<Shape>("dilate", ShapeN(1, 1));

        int batch_size = inputShapes->at(INPUT)->At(0);
        int channel = inputShapes->at(INPUT)->At(1);
        int input_width = inputShapes->at(INPUT)->At(2);
        int input_height = inputShapes->at(INPUT)->At(3);
        const int input_stride = input_height * input_width;

        int output_width = inputShapes->at(SELF_OUT)->At(2);
        int output_height = inputShapes->at(SELF_OUT)->At(3);
        const int output_stride = output_width * output_height;

        const T * pre_grad = Input<T>(PRE_GRAG);
        T * out = Output<T>();
        Value<T>(outputShape->Size(), out, T(0));
        switch (type) {
            case kMax:
            {
                const T *maxIndex = Input<T>(MAX_INDEX);
                for (int i = 0; i < batch_size; ++i) {
                    for (int j = 0; j < channel; ++j) {
                        for (int ph = 0; ph < output_height; ++ph) {
                            for (int pw = 0; pw < output_height; ++pw) {
                                const int output_idx = ph * output_width + pw;
                                const int input_idx = static_cast<int>(maxIndex[output_idx]);
                                out[input_idx] += pre_grad[output_idx];
                            }
                        }
                        out += input_stride;
                        pre_grad += output_stride;
                        maxIndex += output_stride;
                    }

                }
            }
                break;
            case kAvg:
            {
                for (int i = 0; i < batch_size; ++i) {
                    for (int j = 0; j < channel; ++j) {
                        for (int ph = 0; ph < output_height; ++ph) {
                            int hstart = std::max(ph * stride[1] - padding[1], 0);
                            int hend = std::min(hstart + filter[1], input_height);
                            for (int pw = 0; pw < output_width; ++pw) {
                                int wstart = std::max(pw * stride[0] - padding[0], 0);
                                int wend = std::min(wstart + filter[0], input_width);
                                int pool_size = (hend - hstart) * (wend - wstart);
                                T scale = 1.0 / pool_size;
                                for (int h = hstart; h < hend; ++h) {
                                    for (int w = wstart; w < wend; ++w) {
                                        out[h * input_width + w] += (pre_grad[ph * output_width + pw] * scale);
                                    }
                                }
                            }
                        }
                        pre_grad += output_stride;
                        out +=  input_stride;
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

    template <class T, class xpu>
    void PoolingGradOp<T, xpu>::AsyncRun() {
        if (xpu::mode == RunMode::kCpu) {
            Run();
        } else {
            if (!RunOnDevice()) {
                Run();
            }
        }
    }

    template <class T, class xpu>
    bool PoolingGradOp<T, xpu>::RunOnDevice() {
        return false;
    }

    template <class T, class xpu>
    PoolingGradOp<T, xpu>::~PoolingGradOp() {

    }



    void PoolingGradOpProp::InferShape(std::vector<Shape*> &inShape, Shape* outShape) {
        Shape padding = ShapeN(0, 0);
        Shape stride = ShapeN(1, 1);
        Shape dilate = ShapeN(1, 1);
        Shape filter = ShapeN(1, 1);
        if (param->args->count("filter")) {
            auto s = get<Shape>(param->args->at("filter"));
            filter.reShape(s);
//            get<Shape>(param->args->at("filter")).reShape(ShapeN(inShape[2]->At(0), inShape[2]->At(1), filter[0], filter[1]));
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