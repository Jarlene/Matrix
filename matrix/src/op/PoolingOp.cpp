//
// Created by Jarlene on 2017/8/2.
//

#include "matrix/include/op/PoolingOp.h"

namespace matrix {

    template <class T, class Context>
    PoolingOp<T, Context>::PoolingOp(Parameter &param) {
        INIT_PARAMS
    }

    template <class T, class Context>
    bool PoolingOp<T, Context>::Run() {
        PoolType  type = GetArgValue<PoolType>("type", kMax)
        switch (type) {
            case kMax:

                break;
            case kAvg:
                break;
            default:
                Logger::Global()->Fatal("PoolingOp do not support PoolType\n");
                break;
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
    Operator* CreateOp<CPU>(PoolingParam &param, long *size) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new PoolingOp<DType, CPU>(param);
            *size = sizeof(DType) * param.outShapes->Size();
        })
        return op;
    }

    template <>
    Operator* CreateOp<GPU>(PoolingParam &param, long *size) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new PoolingOp<DType, GPU>(param);
            *size = sizeof(DType) * param.outShapes->Size();
        })
        return op;
    }

    PoolingParam::PoolingParam(MatrixType matrixType) : Parameter(matrixType) {

    }

    PoolingOpProp::PoolingOpProp() {
        param = new PoolingParam(kFloat);
    }

    PoolingOpProp::PoolingOpProp(const MatrixType &type) {
        param = new PoolingParam(type);
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
        outShape->reShape(ShapeN(w, h));
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

    void PoolingOpProp::SwitchType(const MatrixType &type) {
        this->param->type = type;
    }
}