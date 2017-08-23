//
// Created by Jarlene on 2017/8/2.
//

#include "matrix/include/op/LossOp.h"


namespace matrix {

    template <class T, class Context>
    LossOp<T, Context>::LossOp(Parameter &param) {
        INIT_PARAMS
    }

    template <class T, class Context>
    bool LossOp<T, Context>::Run() {
        auto lossModel = LossMode::kCrossEntropy;
        if (HasArg("type")) {
            lossModel = GetArgValue<LossMode>("type");
        }
        Tensor<T> data = Inputs()[DATA].template GeneratorTensor<T>(inputShapes[DATA]);
        Tensor<T> label = Inputs()[LABEL].template GeneratorTensor<T>(inputShapes[LABEL]);
        Tensor<T> out = Outputs()[OUT]. template GeneratorTensor<T>(outputShapes[OUT]);
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
    void LossOp<T, Context>::AsyncRun() {
        if (Context::mode == RunMode::kCpu) {
            Run();
        } else {
            if (!RunOnDevice()) {
                Run();
            }
        }
    }

    template <class T, class Context>
    LossOp<T, Context>::~LossOp() {

    }

    template <class T, class Context>
    bool LossOp<T, Context>::RunOnDevice() {
        return false;
    }




    template <>
    Operator* CreateOp<CPU>(LossParam &param, long *size) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new LossOp<DType, CPU>(param);
            int shape = 0;
            for (auto s : param.outShapes) {
                shape += s->Size();
            }
            *size = sizeof(DType) * shape;
        })
        return op;
    }

    template <>
    Operator* CreateOp<GPU>(LossParam &param, long *size) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new LossOp<DType, GPU>(param);
            int shape = 0;
            for (auto s : param.outShapes) {
                shape += s->Size();
            }
            *size = sizeof(DType) * shape;
        })
        return op;
    }

    LossParam::LossParam(MatrixType matrixType) : Parameter(matrixType) {

    }

    LossOpProp::LossOpProp() {
        param = new LossParam(kFloat);
    }

    LossOpProp::LossOpProp(const MatrixType &type) {
        param = new LossParam(type);
    }

    LossOpProp::~LossOpProp() {
        delete param;
    }

    void LossOpProp::InferShape(std::vector<Shape> &inShape, std::vector<Shape*> &outShape) {

    }

    Operator *LossOpProp::CreateOperator(Context context, std::vector<Blob> &input, std::vector<Blob> &output,
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

    void LossOpProp::SwitchType(const MatrixType &type) {
        this->param->type = type;
    }
}