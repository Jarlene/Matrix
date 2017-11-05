//
// Created by Jarlene on 2017/11/5.
//

#ifndef MATRIX_CONVOLUTION1DGRADOP_H
#define MATRIX_CONVOLUTION1DGRADOP_H

#include "Operator.h"
namespace matrix {

    struct Convolution1DGradParam : public Parameter {
        Convolution1DGradParam(MatrixType matrixType) : Parameter(matrixType) {

        }
    };



    template<class T, class Context>
    class Convolution1DGradOp : public Operator {
    SAME_FUNCTION(Convolution1DGrad);
    DISABLE_COPY_AND_ASSIGN(Convolution1DGrad);
        INPUT_TAG(PRE_GRAG, SELF_OUT, DATA, KERNEL, BIAS, COLBUFFER);
    };


    template <typename Context>
    Operator* CreateOp(Convolution1DGradParam &param, long *size);


    class Convolution1DGradOpProp : public OperatorProperty {
    public:
        Convolution1DGradOpProp();
        Convolution1DGradOpProp(const MatrixType &type);
        ~Convolution1DGradOpProp();
        virtual void InferShape(std::vector<Shape*> &inShape, Shape *outShape);
        virtual Operator* CreateOperator(Context context, std::vector<Blob*> &input, Blob* output,
                                         std::vector<Shape*> &inShape, Shape *outShape,
                                         std::map<std::string, Any> &args)  ;
        virtual void SwitchType(const MatrixType &type);
    private:
        Convolution1DGradParam* param;
    };

}

REGISTER_OP_PROPERTY(grad_convolution1d, Convolution1DGradOpProp);
#endif //MATRIX_CONVOLUTION1DGRADOP_H
