//
// Created by Jarlene on 2017/8/23.
//


#include "matrix/include/op/FullConnectedGradOp.h"

namespace matrix {

    template <class T, class Context>
    FullConnectedGradOp<T, Context>::FullConnectedGradOp(Parameter &param) {
        INIT_PARAMS
    }

    template <class T, class Context>
    bool FullConnectedGradOp<T, Context>::Run() {
        if (!HasArg("input_idx")) {
            Logger::Global()->Fatal("FullConnectedGradOp not support. \n");
        }
        int idx = GetArgValue<int>("input_idx");
        Tensor<T> pre_grad = Inputs()[PRE_GRAD]. template GeneratorTensor<T>(inputShapes[PRE_GRAD]);
        switch (idx + 2) {
            case DATA:


                break;
            case WEIGHT:
                break;
            case BIAS:
                break;
            default:
                Logger::Global()->Fatal("FullConnectedGradOp not support. \n");
                break;
        }
        return true;
    }

    template <class T, class Context>
    void FullConnectedGradOp<T, Context>::AsyncRun() {
        if (Context::mode == RunMode::kCpu) {
            Run();
        } else {
            if (!RunOnDevice()) {
                Run();
            }
        }
    }

    template <class T, class Context>
    FullConnectedGradOp<T, Context>::~FullConnectedGradOp() {

    }

    template <class T, class Context>
    bool FullConnectedGradOp<T, Context>::RunOnDevice() {
        return false;
    }




    template <>
    Operator* CreateOp<CPU>(FullConnectedGradParam &param, long *size) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
                op = new FullConnectedGradOp<DType, CPU>(param);
                int shape = 0;
                for (auto s : param.outShapes) {
                    shape += s->Size();
                }
                *size = sizeof(DType) * shape;
        })
        return op;
    }

    template <>
    Operator* CreateOp<GPU>(FullConnectedGradParam &param, long *size) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
                op = new FullConnectedGradOp<DType, GPU>(param);
                int shape = 0;
                for (auto s : param.outShapes) {
                    shape += s->Size();
                }
                *size = sizeof(DType) * shape;
        })
        return op;
    }


    FullConnectedGradOpProp::FullConnectedGradOpProp() {
        param = new FullConnectedGradParam(kFloat);
    }

    FullConnectedGradOpProp::FullConnectedGradOpProp(const MatrixType &type) {
        param = new FullConnectedGradParam(type);
    }

    FullConnectedGradOpProp::~FullConnectedGradOpProp() {
        delete param;
    }

    void FullConnectedGradOpProp::InferShape(std::vector<Shape> &inShape, std::vector<Shape*> &outShape) {
        if (param->args.count("input_idx")) {
            int idx = get<int>(param->args["input_idx"]);
            outShape.at(0)->reShape(inShape.at(idx + 2));
        }

    }

    Operator *FullConnectedGradOpProp::CreateOperator(Context context, std::vector<Blob> &input, std::vector<Blob> &output,
                                                  std::vector<Shape> &inShape, std::vector<Shape*> &outShape,
                                                  std::map<std::string, Any> &args) {

        param->args = args;
        param->inputs = input;
        param->outputs = output;
        InferShape(inShape, outShape);
        param->inputShapes = inShape;
        param->outShapes = outShape;
        BIND_DISPATCH(CreateOp, *param, &memorySize);
    }

    void FullConnectedGradOpProp::SwitchType(const MatrixType &type) {
        this->param->type = type;
    }

}