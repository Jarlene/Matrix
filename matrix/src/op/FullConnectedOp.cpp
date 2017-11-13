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
        if (input.size() == 2) {
            Tensor<T> data(Input<T>(DATA), *inputShapes.at(DATA));
            Tensor<T> weight(Input<T>(WEIGHT), *inputShapes.at(WEIGHT));
            Tensor<T> out(Output<T>(), *outputShape);
            MatrixMul<T>(data, false, weight, false, out);
        } else if (input.size() == 3) {
            Tensor<T> data(Input<T>(DATA), *inputShapes.at(DATA));
            Tensor<T> weight(Input<T>(WEIGHT), *inputShapes.at(WEIGHT));
            Tensor<T> bias(Input<T>(BIAS), *inputShapes.at(BIAS));
            Tensor<T> out(Output<T>(), *outputShape);
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

    Operator *FullConnectedOpProp::CreateOperator(Context context, std::vector<void *> &input, void *output,
                                                  std::vector<Shape *> &inShape, Shape *outShape,
                                                  std::map<std::string, Any> &args) {
        param->args = &args;
        param->output = output;
        InferShape(inShape, outShape);
        for(auto it = inShape.begin(); it != inShape.end(); ++it) {
            param->inputShapes.push_back(*it);
        }
        for(auto it = input.begin(); it != input.end(); ++it) {
            param->inputs.push_back(*it);
        }
        param->outShape = outShape;
        CREATE_OPERATOR(context, param, FullConnectedOp, {
            memorySize = sizeof(DType) * param->outShape->Size();
        })
    }


}