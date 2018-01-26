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
        if (InputSize() < 2) {
            Logger::Global()->Fatal("input shape size less then 2 \n");
        }
        Tensor<T> pre_grad(Input<T>(PRE_GRAD), *inputShapes->at(PRE_GRAD));
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





    void AddGradOpProp::InferShape(std::vector<Shape*> &inShape, Shape *outShape) {
        int idx = param->GetArgValue<int>("input_idx");
        outShape->reShape(*inShape.at(idx + 2));
    }



    INIT_OPERATOR_PROPERTY_CREATE(AddGradOpProp, AddGradOp, true);



}