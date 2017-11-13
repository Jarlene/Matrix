//
// Created by  Jarlene on 2017/8/9.
//

#include "matrix/include/op/UpdateOp.h"

namespace matrix {

    template <class T, class xpu>
    UpdateOp<T, xpu>::UpdateOp(Parameter &param) {
        INIT_PARAMS
    }


    template <class T, class xpu>
    bool UpdateOp<T, xpu>::Run() {
        if (inputShapes.size() < 2) {
            Logger::Global()->Fatal("input shape size less then 2 \n");
        }

        return true;

    }

    template <class T, class xpu>
    void UpdateOp<T, xpu>::AsyncRun() {
        if (xpu::mode == RunMode::kCpu) {
            Run();
        } else if (xpu::mode == RunMode::kGpu) {
            if (!RunOnDevice()) {
                Run();
            }
        }
    }

    template <class T, class xpu>
    UpdateOp<T, xpu>::~UpdateOp() {

    }

    template <class T, class xpu>
    bool UpdateOp<T, xpu>::RunOnDevice() {
        return false;
    }



    UpdateOpProp::~UpdateOpProp() {
        delete param;
    }

    UpdateOpProp::UpdateOpProp(const MatrixType &type)  {
        param = new Parameter(type);
    }

    UpdateOpProp::UpdateOpProp() {
        param = new Parameter(MatrixType::kFloat);
    }

    void UpdateOpProp::InferShape(std::vector<Shape*> &inShape, Shape* outShape) {
    }

    Operator *UpdateOpProp::CreateOperator(Context context, std::vector<void *> &input, void *output,
                                           std::vector<Shape *> &inShape, Shape *outShape,
                                           std::map<std::string, Any> &args) {
        param->args = &args;
        param->inputs = input;
        param->output = output;
        InferShape(inShape, outShape);
        param->inputShapes = inShape;
        param->outShape = outShape;

        CREATE_OPERATOR(param, UpdateOp)
    }



}