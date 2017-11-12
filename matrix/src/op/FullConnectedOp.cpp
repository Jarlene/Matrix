//
// Created by Jarlene on 2017/8/2.
//

#include "matrix/include/op/FullConnectedOp.h"

namespace matrix {

    template <class T, class Context>
    FullConnectedOp<T, Context>::FullConnectedOp(Parameter &param) {
        INIT_PARAMS
    }

    template <class T, class Context>
    bool FullConnectedOp<T, Context>::Run() {
        if (Inputs().size() == 2) {
            Tensor<T> data = Inputs()[DATA]-> template GeneratorTensor<T>(inputShapes.at(DATA));
            Tensor<T> weight = Inputs()[WEIGHT]-> template GeneratorTensor<T>(inputShapes.at(WEIGHT));
            Tensor<T> out = Outputs()-> template GeneratorTensor<T>(outputShapes);
            MatrixMul<T>(data, false, weight, false, out);
        } else if (Inputs().size() == 3) {
            Tensor<T> data = Inputs()[DATA]-> template GeneratorTensor<T>(inputShapes.at(DATA));
            Tensor<T> weight = Inputs()[WEIGHT]-> template GeneratorTensor<T>(inputShapes.at(WEIGHT));
            Tensor<T> bias = Inputs()[BIAS]-> template GeneratorTensor<T>(inputShapes.at(BIAS));
            Tensor<T> out = Outputs()-> template GeneratorTensor<T>(outputShapes);
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

    template <class T, class Context>
    void FullConnectedOp<T, Context>::VariableNode(std::function<void(std::initializer_list<Shape *> shapes)> func) {
        if (inputShapes.size() == 1) {
            Shape weight;
            int rank = inputShapes[0]->Rank();
            for (int i = 0; i < rank - 2; ++i) {
                weight.Append(inputShapes[0]->At(i));
            }
            weight.Append(inputShapes[0]->At(rank - 1));
            if (!HasArg("hide_num")) {
                Logger::Global()->Fatal("FullConnectedOp can not find hide_num params");
            }
            weight.Append(GetArgValue<int>("hide_num"));
            if(HasArg("with_bias")) {
                Shape bias;
                bias.Append(inputShapes[0]->At(0));
                func({&weight, &bias});
            } else {
                func({&weight});
            }
        }
    };



    FullConnectedOpProp::FullConnectedOpProp() {
        param = new Parameter(kFloat);
    }

    FullConnectedOpProp::FullConnectedOpProp(const MatrixType &type) {
        param = new Parameter(type);
    }

    FullConnectedOpProp::~FullConnectedOpProp() {
        delete param;
    }

    void FullConnectedOpProp::InferShape(std::vector<Shape*> &inShape, Shape *outShape) {
        if (inShape.size() == 1) {
            return;
        }
        assert(inShape.size() >= 2);
        assert(outShape != nullptr);
        ProduceMulOpShape(inShape, outShape);
    }

    Operator *FullConnectedOpProp::CreateOperator(Context context, std::vector<Blob*> &input, Blob* output,
                                                  std::vector<Shape*> &inShape, Shape *outShape,
                                                  std::map<std::string, Any> &args) {
        param->args = &args;
        param->inputs = input;
        param->outputs = output;
        InferShape(inShape, outShape);
        param->inputShapes = inShape;
        param->outShapes = outShape;
        CREATE_OPERATOR(param, FullConnectedOp, {
            memorySize = sizeof(DType) * param->outShapes->Size();
        })
    }


}