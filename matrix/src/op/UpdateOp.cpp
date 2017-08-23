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




    template <>
    Operator* CreateOp<CPU>(UpdateParam &param) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new UpdateOp<DType, CPU>(param);
        })
        return op;
    }

    template <>
    Operator* CreateOp<GPU>(UpdateParam &param) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new UpdateOp<DType, GPU>(param);
        })
        return op;
    }

    UpdateOpProp::~UpdateOpProp() {
        delete param;
    }

    UpdateOpProp::UpdateOpProp(const MatrixType &type)  {
        param = new UpdateParam(type);
    }

    UpdateOpProp::UpdateOpProp() {
        param = new UpdateParam(MatrixType::kFloat);
    }

    void UpdateOpProp::InferShape(std::vector<Shape> &inShape, std::vector<Shape> &outShape) {
        outShape.at(0).reShape(inShape.at(0));
    }

    Operator *UpdateOpProp::CreateOperator(Context context, std::vector<Blob> &input, std::vector<Blob> &output,
                                        std::vector<Shape> &inShape, std::vector<Shape> &outShape,
                                        std::map<std::string, Any> &args) {
        param->args = args;
        param->inputs = input;
        param->outputs = output;
        InferShape(inShape, outShape);
        param->inputShapes = inShape;
        param->outShapes = outShape;
        BIND_DISPATCH(CreateOp, *param);
    }

    void UpdateOpProp::SwitchType(const MatrixType &type) {
        this->param->type = type;
    }


}