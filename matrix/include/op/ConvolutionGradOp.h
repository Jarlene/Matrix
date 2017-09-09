//
// Created by Jarlene on 2017/9/8.
//

#ifndef MATRIX_CONVOLUTIONGRADOP_H
#define MATRIX_CONVOLUTIONGRADOP_H


#include "Operator.h"

namespace matrix {

    struct ConvolutionGradParam : public Parameter {
        ConvolutionGradParam(MatrixType matrixType) : Parameter(matrixType) {

        }
    };

    template <class T, class Context>
    class ConvolutionGradOp : public Operator {
    SAME_FUNCTION(ConvolutionGrad);
    DISABLE_COPY_AND_ASSIGN(ConvolutionGrad);
        INPUT_TAG(PRE_GRAG, SELF_OUT, DATA, KERNEL, BIAS, COLBUFFER);
    };

    template <typename Context>
    Operator* CreateOp(ConvolutionGradParam &param, long *size);

    class ConvolutionOpGradProp : public OperatorProperty {
    public:
        ConvolutionOpGradProp();
        ConvolutionOpGradProp(const MatrixType &type);
        ~ConvolutionOpGradProp();
        virtual void InferShape(std::vector<Shape*> &inShape, Shape *outShape);
        virtual Operator* CreateOperator(Context context, std::vector<Blob*> &input, Blob* output,
                                         std::vector<Shape*> &inShape, Shape *outShape,
                                         std::map<std::string, Any> &args)  ;
        virtual void SwitchType(const MatrixType &type);
    private:
        ConvolutionGradParam* param;
    };

}

REGISTER_OP_PROPERTY(grad_convolution, ConvolutionOpGradProp);

#endif //MATRIX_CONVOLUTIONGRADOP_H
