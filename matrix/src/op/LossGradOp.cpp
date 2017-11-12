//
// Created by Jarlene on 2017/11/5.
//

#include "matrix/include/op/LossGradOp.h"



namespace matrix {

    template <class T, class Context>
    LossGradOp<T, Context>::LossGradOp(Parameter &param) {
        INIT_PARAMS
    }

    template <class T, class Context>
    bool LossGradOp<T, Context>::Run() {
        auto lossModel = LossMode::kCrossEntropy;
        if (HasArg("type")) {
            lossModel = GetArgValue<LossMode>("type");
        }
        Tensor<T> data = Inputs()[DATA]->template GeneratorTensor<T>(inputShapes[DATA]);
        Tensor<T> label = Inputs()[LABEL]->template GeneratorTensor<T>(inputShapes[LABEL]);
        Tensor<T> out = Outputs()-> template GeneratorTensor<T>(outputShapes);
        if (lossModel == LossMode::kCrossEntropy) {
            CrossEntropy<T>(data, label, out);
        } else if (lossModel == LossMode::kMSE) {
            RMSLoss<T>(data, label, out);
        } else {
            Logger::Global()->Fatal("LossOp not support other loss.\n");
        }
        return true;
    }

    template <class T, class Context>
    void LossGradOp<T, Context>::AsyncRun() {
        if (Context::mode == RunMode::kCpu) {
            Run();
        } else {
            if (!RunOnDevice()) {
                Run();
            }
        }
    }

    template <class T, class Context>
    LossGradOp<T, Context>::~LossGradOp() {

    }

    template <class T, class Context>
    bool LossGradOp<T, Context>::RunOnDevice() {
        return false;
    }




    template <>
    Operator* CreateOp<CPU>(Parameter &param, long *size) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new LossGradOp<DType, CPU>(param);
            *size = sizeof(DType) * param.outShapes->Size();
        })
        return op;
    }

    template <>
    Operator* CreateOp<GPU>(Parameter &param, long *size) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new LossGradOp<DType, GPU>(param);
            *size = sizeof(DType) * param.outShapes->Size();
        })
        return op;
    }



    LossGradOpProp::LossGradOpProp() {
        param = new Parameter(kFloat);
    }

    LossGradOpProp::LossGradOpProp(const MatrixType &type) {
        param = new Parameter(type);
    }

    LossGradOpProp::~LossGradOpProp() {
        delete param;
    }

    void LossGradOpProp::InferShape(std::vector<Shape*> &inShape, Shape *outShape) {
        assert(outShape != nullptr);
        outShape->Append(1);
    }

    Operator *LossGradOpProp::CreateOperator(Context context, std::vector<Blob*> &input, Blob* output,
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


}