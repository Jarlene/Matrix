//
// Created by Jarlene on 2017/8/2.
//

#include "matrix/include/op/DropoutOp.h"

namespace matrix {

    template <class T, class Context>
    DropoutOp<T, Context>::DropoutOp(Parameter &param) {
        INIT_PARAMS
    }

    template <class T, class Context>
    bool DropoutOp<T, Context>::Run() {
        return Operator::Run();
    }

    template <class T, class Context>
    void DropoutOp<T, Context>::AsyncRun() {
        Operator::AsyncRun();
    }

    template <class T, class Context>
    bool DropoutOp<T, Context>::RunOnDevice() {
        return false;
    }

    template <class T, class Context>
    DropoutOp<T, Context>::~DropoutOp() {

    }




    DropoutOpProp::DropoutOpProp() {
        param = new Parameter(kFloat);
    }

    DropoutOpProp::DropoutOpProp(const MatrixType &type) {
        param = new Parameter(type);
    }

    DropoutOpProp::~DropoutOpProp() {
        delete param;
    }

    void DropoutOpProp::InferShape(std::vector<Shape*> &inShape, Shape* outShape) {

    }

    Operator *DropoutOpProp::CreateOperator(Context context, std::vector<void *> &input, void *output,
                                            std::vector<Shape *> &inShape, Shape *outShape,
                                            std::map<std::string, Any> &args) {
        param->args = &args;
        param->inputs = input;
        param->output = output;
        InferShape(inShape, outShape);
        param->inputShapes = inShape;
        param->outShape = outShape;

        CREATE_OPERATOR(param, DropoutOp, {
            memorySize = sizeof(DType) * param->outShape->Size();
        })
    }


}