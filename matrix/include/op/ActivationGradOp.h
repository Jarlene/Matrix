//
// Created by Jarlene on 2017/8/23.
//

#ifndef MATRIX_ACTIVATIONGRADOP_H
#define MATRIX_ACTIVATIONGRADOP_H

#include "Operator.h"

namespace matrix {


    struct ActivationGradParam : public Parameter {
        ActivationGradParam(MatrixType matrixType) : Parameter(matrixType) {

        }
    };


    template <class T, class Context>
    class ActivationGradOp : public Operator {
        SAME_FUNCTION(ActivationGrad);
        DISABLE_COPY_AND_ASSIGN(ActivationGrad);
        INPUT_TAG(PRE_GRAD, OUT, INPUT);
    };



    class ActivationOpGradProp : public OperatorProperty {
    public:
        ActivationOpGradProp();
        explicit ActivationOpGradProp(const MatrixType &type);
        ~ActivationOpGradProp();
        virtual void InferShape(std::vector<Shape*> &inShape, Shape* outShape);
        virtual Operator* CreateOperator(Context context, std::vector<Blob*> &input, Blob* output,
                                         std::vector<Shape*> &inShape, Shape* outShape,
                                         std::map<std::string, Any> &args) ;
        virtual void SwitchType(const MatrixType &type);
    private:
        ActivationGradParam* param;
    };



    template <typename xpu>
    Operator* CreateOp(ActivationGradParam &param, long *size);

}

REGISTER_OP_PROPERTY(grad_activation, ActivationOpGradProp);


#endif //MATRIX_ACTIVATIONGRADOP_H
