//
// Created by Jarlene on 2017/11/5.
//

#include "matrix/include/op/FlattenOp.h"

namespace matrix {


    template <class T, class Context>
    FlattenOp<T, Context>::FlattenOp(Parameter &param) {
        INIT_PARAMS
    }

    template <class T, class Context>
    bool FlattenOp<T, Context>::Run() {
        FallThrow();
        return true;
    }

    template <class T, class Context>
    void FlattenOp<T, Context>::AsyncRun() {
        if (Context::mode == RunMode::kCpu) {
            Run();
        } else {
            if (!RunOnDevice()) {
                Run();
            }
        }
    }

    template <class T, class Context>
    bool FlattenOp<T, Context>::RunOnDevice() {
        return false;
    }

    template <class T, class Context>
    FlattenOp<T, Context>::~FlattenOp() {

    }


    template <>
    Operator* CreateOp<CPU>(Parameter &param, long* size) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new FlattenOp<DType, CPU>(param);
            *size = 0;
        })
        return op;
    }

    template <>
    Operator* CreateOp<GPU>(Parameter &param, long *size) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new FlattenOp<DType, GPU>(param);
            *size = 0;
        })
        return op;
    }

    FlattenOpProp::FlattenOpProp() {
        param = new Parameter(kFloat);
    }

    FlattenOpProp::FlattenOpProp(const MatrixType &type) {
        param = new Parameter(type);
    }

    FlattenOpProp::~FlattenOpProp() {
        delete param;
    }

    void FlattenOpProp::InferShape(std::vector<Shape *> &inShape, Shape *outShape) {
        outShape->reShape(ShapeN(inShape[0]->At(0), inShape[0]->StrideExclude(0)));
    }

    Operator *FlattenOpProp::CreateOperator(Context context, std::vector<Blob *> &input, Blob *output,
                                            std::vector<Shape *> &inShape, Shape *outShape,
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

