//
// Created by 郑珊 on 2017/8/9.
//

#include "matrix/include/op/ApplyGradOp.h"

namespace matrix {

    template <class T, class xpu>
    ApplyGradOp<T, xpu>::ApplyGradOp(ApplyGradParam &param) {
        this->inputShapes = param.inputShapes;
        this->outputShapes = param.outShapes;
        this->input.insert(input.end(), param.inputs.begin(), param.inputs.end());
        for (Blob b : param.outputs) {
            this->output.push_back(b);
        }
        this->args = param.args;
    }


    template <class T, class xpu>
    bool ApplyGradOp<T, xpu>::Run() {
        if (inputShapes.size() < 2) {
            Logger::Global()->Fatal("input shape size less then 2 \n");
        }

        return true;

    }

    template <class T, class xpu>
    void ApplyGradOp<T, xpu>::AsyncRun() {
        if (xpu::mode == RunMode::kCpu) {
            Run();
        } else if (xpu::mode == RunMode::kGpu) {
            if (!RunOnDevice()) {
                Run();
            }
        }
    }

    template <class T, class xpu>
    ApplyGradOp<T, xpu>::~ApplyGradOp() {

    }

    template <class T, class xpu>
    bool ApplyGradOp<T, xpu>::RunOnDevice() {
        return false;
    }




    template <>
    Operator* CreateOp<CPU>(ApplyGradParam &param) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new ApplyGradOp<DType, CPU>(param);
        })
        return op;
    }

    template <>
    Operator* CreateOp<GPU>(ApplyGradParam &param) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new ApplyGradOp<DType, GPU>(param);
        })
        return op;
    }

    ApplyGradProp::~ApplyGradProp() {
        delete param;
    }

    ApplyGradProp::ApplyGradProp(const MatrixType &type)  {
        param = new ApplyGradParam(type);
    }

    ApplyGradProp::ApplyGradProp() {
        param = new ApplyGradParam(MatrixType::kFloat);
    }

    Operator *ApplyGradProp::CreateOperator(Context context, std::vector<Blob> &input, std::vector<Blob> &output,
                                        std::vector<Shape> &inShape, std::vector<Shape> &outShape,
                                        std::map<std::string, Any> &args) {
        InferShape(inShape, outShape);
        param->outputs = output;
        param->inputs = input;
        param->inputShapes = inShape;
        param->outShapes = outShape;
        param->args = args;
        BIND_DISPATCH(CreateOp, *param);
    }

    void ApplyGradProp::InferShape(std::vector<Shape> &inShape, std::vector<Shape> &outShape) {
        outShape.at(0).reShape(inShape.at(0));
    }


}