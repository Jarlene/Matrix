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
        Tensor<T> pre_grad = Inputs()[PRE_GRAD]. template GeneratorTensor<T>(inputShapes[PRE_GRAD]);
        Tensor<T> out_grad = Outputs()[OUT_GRAD]. template GeneratorTensor<T>(outputShapes[OUT_GRAD]);
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




    template <>
    Operator* CreateOp<CPU>(AddGradParam &param, long *size) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new AddGradOp<DType, CPU>(param);
            int shape = 0;
            for (Shape s : param.outShapes) {
                shape += s.Size();
            }
            *size = sizeof(DType) * shape;
        })
        return op;
    }

    template <>
    Operator* CreateOp<GPU>(AddGradParam &param, long *size) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new AddGradOp<DType, GPU>(param);
            int shape = 0;
            for (Shape s : param.outShapes) {
                shape += s.Size();
            }
            *size = sizeof(DType) * shape;
        })
        return op;
    }

    AddGradOpProp::AddGradOpProp(const MatrixType &type)  {
        param = new AddGradParam(type);
    }

    AddGradOpProp::AddGradOpProp() {
        param = new AddGradParam(MatrixType::kFloat);
    }

    Operator *AddGradOpProp::CreateOperator(Context context, std::vector<Blob> &input, std::vector<Blob> &output,
                                        std::vector<Shape> &inShape, std::vector<Shape> &outShape,
                                        std::map<std::string, Any> &args) {
        // attention order
        param->outputs = output;
        param->inputs = input;
        param->args = args;
        InferShape(inShape, outShape);
        param->inputShapes = inShape;
        param->outShapes = outShape;
        BIND_DISPATCH(CreateOp, *param, &memorySize);
    }

    void AddGradOpProp::InferShape(std::vector<Shape> &inShape, std::vector<Shape> &outShape) {
        if (param->args.count("input_idx")) {
            int idx = get<int>(param->args["input_idx"]);
            outShape.at(0).reShape(inShape.at(idx + 2));
        }
    }

    void AddGradOpProp::SwitchType(const MatrixType &type) {
        param->type = type;
    }

    AddGradOpProp::~AddGradOpProp() {
        delete param;
    }

}