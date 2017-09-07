//
// Created by Jarlene on 2017/9/6.
//

#ifndef MATRIX_CONVOLUTION1DOP_H
#define MATRIX_CONVOLUTION1DOP_H

#include "Operator.h"

namespace matrix {

    struct Convolution1DParam : public Parameter {
        Convolution1DParam(MatrixType matrixType) : Parameter(matrixType) {

        }
    };

    template <class T, class Context>
    class Convolution1DOp : public Operator {
    SAME_FUNCTION(Convolution1D);
    DISABLE_COPY_AND_ASSIGN(Convolution1D);
        INPUT_TAG(DATA, KERNEL, BIAS);
    };

    template <typename Context>
    Operator* CreateOp(Convolution1DParam &param, long *size);

    class Convolution1DOpProp : public OperatorProperty {
    public:
        Convolution1DOpProp();
        Convolution1DOpProp(const MatrixType &type);
        ~Convolution1DOpProp();
        virtual void InferShape(std::vector<Shape*> &inShape, Shape *outShape);
        virtual Operator* CreateOperator(Context context, std::vector<Blob*> &input, Blob* output,
                                         std::vector<Shape*> &inShape, Shape *outShape,
                                         std::map<std::string, Any> &args)  ;
        virtual void SwitchType(const MatrixType &type);
    private:
        Convolution1DParam* param;
    };

}

REGISTER_OP_PROPERTY(convolution1d, Convolution1DOpProp);

#endif //MATRIX_CONVOLUTION1DOP_H
