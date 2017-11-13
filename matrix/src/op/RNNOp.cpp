//
// Created by Jarlene on 2017/11/9.
//

#include "matrix/include/op/RNNOp.h"

namespace matrix {


    template <class T, class Context>
    RNNOp<T, Context>::RNNOp(Parameter &param) {
        INIT_PARAMS
    }

    template <class T, class Context>
    bool RNNOp<T, Context>::Run() {
        return Operator::Run();
    }

    template <class T, class Context>
    void RNNOp<T, Context>::AsyncRun() {
        Operator::AsyncRun();
    }

    template <class T, class Context>
    bool RNNOp<T, Context>::RunOnDevice() {
        return false;
    }

    template <class T, class Context>
    RNNOp<T, Context>::~RNNOp() {

    }





    RNNOpProp::RNNOpProp() {
        param = new Parameter(kFloat);
    }

    RNNOpProp::RNNOpProp(const MatrixType &type) {
        param = new Parameter(type);
    }

    RNNOpProp::~RNNOpProp() {
        delete param;
    }

    void RNNOpProp::InferShape(std::vector<Shape*> &inShape, Shape *outShape) {

    }

    Operator *RNNOpProp::CreateOperator(Context context, std::vector<void *> &input, void *output,
                                        std::vector<Shape *> &inShape, Shape *outShape,
                                        std::map<std::string, Any> &args) {
        param->args = &args;
        param->output = output;
        InferShape(inShape, outShape);
        for(auto it = inShape.begin(); it != inShape.end(); ++it) {
            param->inputShapes.push_back(*it);
        }
        for(auto it = input.begin(); it != input.end(); ++it) {
            param->inputs.push_back(*it);
        }
        param->outShape = outShape;
        CREATE_OPERATOR(context, param, RNNOp, {
            memorySize = sizeof(DType) * param->outShape->Size();
        })
    }


}
