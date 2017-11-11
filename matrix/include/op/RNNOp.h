//
// Created by Jarlene on 2017/11/9.
//

#ifndef MATRIX_RNNOP_H
#define MATRIX_RNNOP_H


#include "Operator.h"

namespace matrix {

    struct RNNParam : public Parameter {
        RNNParam(MatrixType matrixType) : Parameter(matrixType) {};
    };
    
    template <class T, class Context>
    class RNNOp : public Operator {
    SAME_FUNCTION(RNN);
    DISABLE_COPY_AND_ASSIGN(RNN);
    };


    template <class T>
    class RNNState : public State {
    public:

    };

    template <typename Context>
    Operator* CreateOp(RNNParam &param, long *size);


    class RNNOpProp : public OperatorProperty {
    public:
        RNNOpProp();
        RNNOpProp(const MatrixType &type);
        ~RNNOpProp();
        virtual void InferShape(std::vector<Shape*> &inShape, Shape *outShape);
        virtual Operator* CreateOperator(Context context, std::vector<Blob*> &input, Blob* output,
                                         std::vector<Shape*> &inShape, Shape *outShape,
                                         std::map<std::string, Any> &args)  ;
        virtual void SwitchType(const MatrixType &type);
    private:
        RNNParam* param;
    };
}


REGISTER_OP_PROPERTY(rnn, RNNOpProp);




#endif //MATRIX_RNNOP_H
