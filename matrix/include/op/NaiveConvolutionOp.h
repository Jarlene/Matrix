//
// Created by Jarlene on 2017/9/6.
//

#ifndef MATRIX_NAVIECONVOLUTIONOP_H
#define MATRIX_NAVIECONVOLUTIONOP_H

#include "Operator.h"

namespace matrix {

    struct NaiveConvolutionParam : public Parameter {
        NaiveConvolutionParam(MatrixType matrixType) : Parameter(matrixType) {

        }
    };

    template <class T, class Context>
    class NaiveConvolutionOp : public Operator {
        SAME_FUNCTION(NaiveConvolution);
        DISABLE_COPY_AND_ASSIGN(NaiveConvolution);
        INPUT_TAG(DATA, KERNEL, BIAS);
    };

    template <typename Context>
    Operator* CreateOp(NaiveConvolutionParam &param, long *size);

    class NaiveConvolutionOpProp : public OperatorProperty {
    public:
        NaiveConvolutionOpProp();
        NaiveConvolutionOpProp(const MatrixType &type);
        ~NaiveConvolutionOpProp();
        virtual void InferShape(std::vector<Shape*> &inShape, Shape *outShape);
        virtual Operator* CreateOperator(Context context, std::vector<Blob*> &input, Blob* output,
                                         std::vector<Shape*> &inShape, Shape *outShape,
                                         std::map<std::string, Any> &args)  ;
        virtual void SwitchType(const MatrixType &type);
    private:
        NaiveConvolutionParam* param;
    };

}

REGISTER_OP_PROPERTY(navie_convolution, NaiveConvolutionOpProp);

#endif //MATRIX_NAVIECONVOLUTIONOP_H
