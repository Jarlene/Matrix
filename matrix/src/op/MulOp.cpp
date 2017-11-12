//
// Created by Jarlene on 2017/8/2.
//

#include "matrix/include/op/MulOp.h"

namespace matrix {


    template <class T, class Context>
    MulOp<T, Context>::MulOp(Parameter &param) {
        INIT_PARAMS
    }

    template <class T, class Context>
    bool MulOp<T, Context>::Run() {
        Tensor<T> data = Inputs()[INPUT1]-> template GeneratorTensor<T>(inputShapes.at(INPUT1));
        Tensor<T> weight = Inputs()[INPUT2]-> template GeneratorTensor<T>(inputShapes.at(INPUT2));
        Tensor<T> out = Outputs()-> template GeneratorTensor<T>(outputShapes);
        MatrixMul<T>(data, false, weight, false, out);
        return true;
    }

    template <class T, class Context>
    void MulOp<T, Context>::AsyncRun() {
        if (Context::mode == RunMode::kCpu) {
            Run();
        } else {
            if (!RunOnDevice()) {
                Run();
            }
        }
    }

    template <class T, class Context>
    bool MulOp<T, Context>::RunOnDevice() {
        return false;
    }

    template <class T, class Context>
    MulOp<T, Context>::~MulOp() {

    }



    template <>
    Operator* CreateOp<CPU>(Parameter &param, long *size) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new MulOp<DType, CPU>(param);
            *size = sizeof(DType) * param.outShapes->Size();
        })
        return op;
    }

    template <>
    Operator* CreateOp<GPU>(Parameter &param, long *size) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new MulOp<DType, GPU>(param);
            *size = sizeof(DType) * param.outShapes->Size();
        })
        return op;
    }


    MulOpProp::MulOpProp() {
        param = new Parameter(kFloat);
    }

    MulOpProp::MulOpProp(const MatrixType &type) {
        param = new Parameter(type);
    }

    MulOpProp::~MulOpProp() {
        delete param;
    }

    void MulOpProp::InferShape(std::vector<Shape*> &inShape, Shape *outShape) {
        assert(inShape.size() >= 2);
        assert(outShape != nullptr);
        ProduceMulOpShape(inShape, outShape);
    }

    Operator *MulOpProp::CreateOperator(Context context, std::vector<Blob*> &input, Blob* output,
                                        std::vector<Shape*> &inShape, Shape *outShape,
                                        std::map<std::string, Any> &args) {
        param->args = &args;
        param->inputs = input;
        param->outputs = output;
        InferShape(inShape, outShape);
        param->inputShapes = inShape;
        param->outShapes = outShape;
        BIND_DISPATCH(CreateOp, *param, &memorySize);
    }


}