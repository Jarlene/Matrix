//
// Created by Jarlene on 2017/7/31.
//

#ifndef MATRIX_CONVOLUTIONOP_H
#define MATRIX_CONVOLUTIONOP_H

#include "Operator.h"

namespace matrix {

    struct ConvolutionParam : public Parameter {
        ConvolutionParam(MatrixType matrixType) : Parameter(matrixType) {

        }
    };

    template <class T, class Context>
    class ConvolutionOp : public Operator {
    SAME_FUNCTION(Convolution);
    DISABLE_COPY_AND_ASSIGN(Convolution);
    };

    template <typename Context>
    Operator* CreateOp(ConvolutionParam &param);

    class ConvolutionOpProp : public OperatorProperty {
    public:
        ConvolutionOpProp();
        ConvolutionOpProp(const MatrixType &type);
        ~ConvolutionOpProp();
        virtual void InferShape(std::vector<Shape> &inShape, std::vector<Shape> &outShape);
        virtual Operator* CreateOperator(Context context, std::vector<Blob> &input, std::vector<Blob> &output,
                                         std::vector<Shape> &inShape, std::vector<Shape> &outShape,
                                         std::map<std::string, Any> &args) ;
    private:
        ConvolutionParam* param;
    };

}

REGISTER_OP_PROPERTY(convolution, ConvolutionOpProp);

#endif //MATRIX_CONVOLUTIONOP_H
