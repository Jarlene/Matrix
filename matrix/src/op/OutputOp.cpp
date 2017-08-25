//
// Created by Jarlene on 2017/8/9.
//

#include "matrix/include/op/OutputOp.h"

namespace matrix {
    template <class T, class Context>
    OutputOp<T, Context>::OutputOp(Parameter &param) {
        INIT_PARAMS
    }

    template <class T, class Context>
    bool OutputOp<T, Context>::Run() {
        auto outModel = OutputMode::kSoftmax;
        if (HasArg("type")) {
            outModel = GetArgValue<OutputMode>("type");
        }
        Tensor<T> data = Inputs()[DATA]-> template GeneratorTensor<T>(inputShapes[DATA]);
        Tensor<T> out = Outputs() -> template GeneratorTensor<T>(outputShapes);
        if (outModel == OutputMode::kSoftmax) {
            Softmax<T>(data, out);
        } else {
            Logger::Global()->Fatal("OutputOp not support other output.\n");
        }
        return true;
    }

    template <class T, class Context>
    void OutputOp<T, Context>::AsyncRun() {
        if (Context::mode == RunMode::kCpu) {
            Run();
        } else {
            if (!RunOnDevice()) {
                Run();
            }
        }
    }

    template <class T, class Context>
    bool OutputOp<T, Context>::RunOnDevice() {
        return false;
    }

    template <class T, class Context>
    OutputOp<T, Context>::~OutputOp() {

    }



    template <>
    Operator* CreateOp<CPU>(OutputParam &param, long *size) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new OutputOp<DType, CPU>(param);
            *size = sizeof(DType) * param.outShapes->Size();
        })
        return op;
    }

    template <>
    Operator* CreateOp<GPU>(OutputParam &param, long *size) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new OutputOp<DType, GPU>(param);
            *size = sizeof(DType) * param.outShapes->Size();
        })
        return op;
    }



    OutputOpProp::OutputOpProp() {
        param = new OutputParam(kFloat);
    }

    OutputOpProp::OutputOpProp(const MatrixType &type) {
        param = new OutputParam(type);
    }

    OutputOpProp::~OutputOpProp() {
        delete param;
    }

    void OutputOpProp::InferShape(std::vector<Shape*> &inShape, Shape *outShape) {
        assert(outShape != nullptr);
        outShape->reShape(*inShape[0]);
    }

    Operator *OutputOpProp::CreateOperator(Context context, std::vector<Blob*> &input, Blob* output,
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

    void OutputOpProp::SwitchType(const MatrixType &type) {
        this->param->type = type;
    }

}