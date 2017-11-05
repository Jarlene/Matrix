//
// Created by Jarlene on 2017/11/5.
//

#ifndef MATRIX_LOSSGRADOP_H
#define MATRIX_LOSSGRADOP_H

#include "Operator.h"


namespace matrix {

    struct LossGradParam : public Parameter {
        LossGradParam(MatrixType matrixType) : Parameter(matrixType) {

        };
    };

    template <class T, class Context>
    class LossGradOp : public Operator{
    SAME_FUNCTION(LossGrad);
    DISABLE_COPY_AND_ASSIGN(LossGrad);
        INPUT_TAG(DATA, LABEL);
    };

    template <typename Context>
    Operator* CreateOp(LossGradParam &param, long *size);


    class LossGradOpProp : public OperatorProperty {
    public:
        LossGradOpProp();
        LossGradOpProp(const MatrixType &type);
        ~LossGradOpProp();
        virtual void InferShape(std::vector<Shape*> &inShape, Shape *outShape);
        virtual Operator* CreateOperator(Context context, std::vector<Blob*> &input, Blob* output,
                                         std::vector<Shape*> &inShape, Shape *outShape,
                                         std::map<std::string, Any> &args)  ;
        virtual void SwitchType(const MatrixType &type);
    private:
        LossGradParam* param;
    };
}


REGISTER_OP_PROPERTY(grad_loss, LossGradOpProp);


#endif //MATRIX_LOSSGRADOP_H
