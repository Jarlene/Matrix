//
// Created by Jarlene on 2017/8/2.
//

#include "matrix/include/op/SubOp.h"

namespace matrix {
    template <class T, class Context>
    SubOp<T, Context>::SubOp(Parameter &param) {
        INIT_PARAMS
    }

    template <class T, class Context>
    bool SubOp<T, Context>::Run() {
        return Operator::Run();
    }

    template <class T, class Context>
    void SubOp<T, Context>::AsyncRun() {
        Operator::AsyncRun();
    }

    template <class T, class Context>
    bool SubOp<T, Context>::RunOnDevice() {
        return false;
    }

    template <class T, class Context>
    SubOp<T, Context>::~SubOp() {

    }



    template <>
    Operator* CreateOp<CPU>(SubParam &param, long *size) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new SubOp<DType, CPU>(param);
            int shape = 0;
            for (auto s : param.outShapes) {
                shape += s->Size();
            }
            *size = sizeof(DType) * shape;
        })
        return op;
    }

    template <>
    Operator* CreateOp<GPU>(SubParam &param, long *size) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new SubOp<DType, GPU>(param);
            int shape = 0;
            for (auto s : param.outShapes) {
                shape += s->Size();
            }
            *size = sizeof(DType) * shape;
        })
        return op;
    }

    SubParam::SubParam(MatrixType matrixType) : Parameter(matrixType) {

    }

    SubOpProp::SubOpProp() {
        param = new SubParam(kFloat);
    }

    SubOpProp::SubOpProp(const MatrixType &type) {
        param = new SubParam(type);
    }

    SubOpProp::~SubOpProp() {
        delete param;
    }

    void SubOpProp::InferShape(std::vector<Shape*> &inShape, std::vector<Shape*> &outShape) {

    }

    Operator *SubOpProp::CreateOperator(Context context, std::vector<Blob*> &input, std::vector<Blob*> &output,
                                        std::vector<Shape*> &inShape, std::vector<Shape*> &outShape,
                                        std::map<std::string, Any> &args) {
        param->args = args;
        param->inputs = input;
        param->outputs = output;
        InferShape(inShape, outShape);
        param->inputShapes = inShape;
        param->outShapes = outShape;
        BIND_DISPATCH(CreateOp, *param, &memorySize);
    }
    void SubOpProp::SwitchType(const MatrixType &type) {
        this->param->type = type;
    }
}