//
// Created by Jarlene on 2017/8/2.
//

#include "matrix/include/store/MemoryManager.h"
#include "matrix/include/op/PoolingOp.h"
#include "matrix/include/op/ReduceOp.h"

namespace matrix {

    template <class T, class xpu>
    PoolingOp<T, xpu>::PoolingOp(Parameter &param) {
        INIT_PARAMS
    }

    template <class T, class xpu>
    bool PoolingOp<T, xpu>::Run() {
        PoolType  type = GetArgValue<PoolType>("type", kMax);


        auto filter = GetArgValue<Shape>("filter");
        auto stride = GetArgValue<Shape>("stride", ShapeN(1, 1));
        auto padding = GetArgValue<Shape>("padding", ShapeN(0 ,0));
        auto dilate = GetArgValue<Shape>("dilate", ShapeN(1, 1));

        int batch_size = inputShapes->at(0)->At(0);
        int channel = inputShapes->at(0)->At(1);
        int input_width = inputShapes->at(0)->At(2);
        int input_height = inputShapes->at(0)->At(3);

        int group = GetArgValue<int>("group" , 1);

        const T* input = Input<T>(DATA);
        T* out = Output<T>();

        if (type == kMax) {
            T * maxIndex = InputNonConst<T>(MAX_INDEX);
            Value(inputShapes->at(MAX_INDEX)->Size() ,maxIndex, T(0));
            pooling2D(input, batch_size, channel/group, input_width, input_height, stride[0], stride[1],
                      padding[0], padding[1],filter[0], filter[1], dilate[0], dilate[1],out, type, maxIndex);
        } else {
            pooling2D(input, batch_size, channel/group, input_width, input_height, stride[0], stride[1],
                      padding[0], padding[1],filter[0], filter[1], dilate[0], dilate[1], out, type);
        }
        return true;
    }

    template <class T, class xpu>
    void PoolingOp<T, xpu>::AsyncRun() {
        if (xpu::mode == RunMode::kCpu) {
            Run();
        } else {
            if (!RunOnDevice()) {
                Run();
            }
        }
    }

    template <class T, class xpu>
    bool PoolingOp<T, xpu>::RunOnDevice() {
        return false;
    }

    template <class T, class xpu>
    PoolingOp<T, xpu>::~PoolingOp() {

    }

    template <class T, class xpu>
    bool PoolingOp<T, xpu>::ShareNodes(std::function<void(std::initializer_list<Shape *> shapes)> func) {
        if (GetArgValue<PoolType>("type", kMax) == kMax && InputSize() == 1) {
            Shape colShape;
            colShape.reShape(*outputShape);
            func({&colShape});
            return true;
        }
        return false;
    }


    void PoolingOpProp::InferShape(std::vector<Shape*> &inShape, Shape* outShape) {
        Shape padding = ShapeN(0, 0);
        Shape stride = ShapeN(1, 1);
        Shape dilate = ShapeN(1, 1);
        Shape filter = ShapeN(1, 1);
        if (param->args->count("filter")) {
            auto s = get<Shape>(param->args->at("filter"));
            filter.reShape(s);
//            get<Shape>(param->args->at("filter")).reShape(ShapeN(inShape[0]->At(0), inShape[0]->At(1), filter[0], filter[1]));
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

    INIT_OPERATOR_PROPERTY_CREATE(PoolingOpProp, PoolingOp, true);

}