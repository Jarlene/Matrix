//
// Created by Jarlene on 2017/8/2.
//

#include "matrix/include/op/GRUOp.h"

namespace matrix {

    template <class T, class Context>
    GRUOp<T, Context>::GRUOp(Parameter &param) {
        INIT_PARAMS
    }

    template <class T, class Context>
    bool GRUOp<T, Context>::Run() {
        return Operator::Run();
    }

    template <class T, class Context>
    void GRUOp<T, Context>::AsyncRun() {
        Operator::AsyncRun();
    }

    template <class T, class Context>
    bool GRUOp<T, Context>::RunOnDevice() {
        return false;
    }

    template <class T, class Context>
    GRUOp<T, Context>::~GRUOp() {

    }






    GRUOpProp::GRUOpProp() {
        param = new Parameter(kFloat);
    }

    GRUOpProp::GRUOpProp(const MatrixType &type) {
        param = new Parameter(type);
    }

    GRUOpProp::~GRUOpProp() {
        delete param;
    }

    void GRUOpProp::InferShape(std::vector<Shape*> &inShape, Shape *outShape) {

    }

    Operator *GRUOpProp::CreateOperator(Context context, std::vector<void *> *input, void *output,
                                        std::vector<Shape *> *inShape, Shape *outShape,
                                        std::map<std::string, Any> &args) {
        param->args = &args;
        param->output = output;
        InferShape(*inShape, outShape);
        param->inputShapes = inShape;
        param->inputs = input;
        param->outShape = outShape;
        CREATE_OPERATOR(context, param, GRUOp, {
            memorySize = sizeof(DType) * param->outShape->Size();
        })
    }


}