//
// Created by Jarlene on 2017/8/23.
//

#include "matrix/include/op/AddGradOp.h"


namespace matrix {


    template <class T, class xpu>
    AddGradOp<T, xpu>::AddGradOp(Parameter &param) {
        INIT_PARAMS
    }


    template <class T, class xpu>
    bool AddGradOp<T, xpu>::Run() {
        if (inputShapes.size() < 2) {
            Logger::Global()->Fatal("input shape size less then 2 \n");
        }
        Tensor<T> pre_grad(Input<T>(PRE_GRAD), *inputShapes[PRE_GRAD]);
        Tensor<T> out_grad(Output<T>(), *outputShape);
        Copy(pre_grad, out_grad);
        return true;

    }

    template <class T, class xpu>
    void AddGradOp<T, xpu>::AsyncRun() {
        if (xpu::mode == RunMode::kCpu) {
            Run();
        } else if (xpu::mode == RunMode::kGpu) {
            if (!RunOnDevice()) {
                Run();
            }
        }
    }

    template <class T, class xpu>
    AddGradOp<T, xpu>::~AddGradOp() {

    }

    template <class T, class xpu>
    bool AddGradOp<T, xpu>::RunOnDevice() {
        return false;
    }



    AddGradOpProp::AddGradOpProp(const MatrixType &type)  {
        param = new Parameter(type);
    }

    AddGradOpProp::AddGradOpProp() {
        param = new Parameter(MatrixType::kFloat);
    }


    void AddGradOpProp::InferShape(std::vector<Shape*> &inShape, Shape *outShape) {
        if (param->args->count("input_idx")) {
            int idx = get<int>(param->args->at("input_idx"));
            outShape->reShape(*inShape.at(idx + 2));
        }
    }

    AddGradOpProp::~AddGradOpProp() {
        delete param;
    }

    Operator *AddGradOpProp::CreateOperator(Context context, std::vector<void *> &input, void *output,
                                            std::vector<Shape *> &inShape, Shape *outShape,
                                            std::map<std::string, Any> &args) {
        // attention order
        param->output = output;
        param->args = &args;
        InferShape(inShape, outShape);
        for(auto it = inShape.begin(); it != inShape.end(); ++it) {
            param->inputShapes.push_back(*it);
        }
        for(auto it = input.begin(); it != input.end(); ++it) {
            param->inputs.push_back(*it);
        }
        param->outShape = outShape;
        CREATE_OPERATOR(context, param, AddGradOp, {
            memorySize = sizeof(DType) * param->outShape->Size();
        })
    }



}