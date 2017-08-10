//
// Created by Jarlene on 2017/8/2.
//

#include "matrix/include/op/FullConnectedOp.h"

namespace matrix {

    template <class T, class Context>
    FullConnectedOp<T, Context>::FullConnectedOp(FullConnectedParam &param) {
        this->output = param.outputs;
        this->outputShapes = param.outShapes;
        this->input = param.inputs;
        this->inputShapes = param.inputShapes;
        this->args = param.args;
    }

    template <class T, class Context>
    bool FullConnectedOp<T, Context>::Run() {
        if (Inputs().size() == 2) {
            Tensor<T> data = Inputs()[DATA]. template GeneratorTensor<T>(inputShapes.at(DATA));
            Tensor<T> weight = Inputs()[WEIGHT]. template GeneratorTensor<T>(inputShapes.at(WEIGHT));
            Tensor<T> out = Outputs()[OUT]. template GeneratorTensor<T>(outputShapes.at(OUT));
            MatrixMul<T>(data, false, weight, false, out);
        } else if (Inputs().size() == 3) {
            Tensor<T> data = Inputs()[DATA]. template GeneratorTensor<T>(inputShapes.at(DATA));
            Tensor<T> weight = Inputs()[WEIGHT]. template GeneratorTensor<T>(inputShapes.at(WEIGHT));
            Tensor<T> bias = Inputs()[BIAS]. template GeneratorTensor<T>(inputShapes.at(BIAS));
            Tensor<T> out = Outputs()[OUT]. template GeneratorTensor<T>(outputShapes.at(OUT));
            MatrixMul<T>(data, false, weight, false, out);
            Add<T>(out, bias, out);
        }
        return true;
    }

    template <class T, class Context>
    void FullConnectedOp<T, Context>::AsyncRun() {
        if (Context::mode == RunMode::kCpu) {
            Run();
        } else {
            if (!RunOnDevice()) {
                Run();
            }
        }
    }

    template <class T, class Context>
    FullConnectedOp<T, Context>::~FullConnectedOp() {

    }

    template <class T, class Context>
    bool FullConnectedOp<T, Context>::RunOnDevice() {
        return false;
    }




    template <>
    Operator* CreateOp<CPU>(FullConnectedParam &param, long *size) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new FullConnectedOp<DType, CPU>(param);
            int shape = 0;
            for (Shape s : param.outShapes) {
                shape += s.Size();
            }
            *size = sizeof(DType) * shape;
        })
        return op;
    }

    template <>
    Operator* CreateOp<GPU>(FullConnectedParam &param, long *size) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new FullConnectedOp<DType, GPU>(param);
            int shape = 0;
            for (Shape s : param.outShapes) {
                shape += s.Size();
            }
            *size = sizeof(DType) * shape;
        })
        return op;
    }

    FullConnectedParam::FullConnectedParam(MatrixType matrixType) : Parameter(matrixType) {

    }

    FullConnectedOpProp::FullConnectedOpProp() {
        param = new FullConnectedParam(kFloat);
    }

    FullConnectedOpProp::FullConnectedOpProp(const MatrixType &type) {
        param = new FullConnectedParam(type);
    }

    FullConnectedOpProp::~FullConnectedOpProp() {
        delete param;
    }

    void FullConnectedOpProp::InferShape(std::vector<Shape> &inShape, std::vector<Shape> &outShape) {
        assert(inShape.size() >= 2);
        assert(outShape.size() >= 1);
        ProduceMulOpShape(inShape, outShape[0]);
    }

    Operator *FullConnectedOpProp::CreateOperator(Context context, std::vector<Blob> &input, std::vector<Blob> &output,
                                                  std::vector<Shape> &inShape, std::vector<Shape> &outShape,
                                                  std::map<std::string, Any> &args) {
        InferShape(inShape, outShape);
        param->inputs = input;
        param->outputs = output;
        param->inputShapes = inShape;
        param->outShapes = outShape;
        param->args = args;
        BIND_DISPATCH(CreateOp, *param, &memorySize);
    }

}