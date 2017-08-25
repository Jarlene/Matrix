//
// Created by Jarlene on 2017/8/23.
//

#ifndef MATRIX_FULLCONNECTEDGRADOP_H
#define MATRIX_FULLCONNECTEDGRADOP_H


#include "Operator.h"

namespace matrix {


    struct  FullConnectedGradParam : public Parameter {
        FullConnectedGradParam(MatrixType matrixType) : Parameter(matrixType) {

        }
    };

    template <class T, class Context>
    class FullConnectedGradOp : public Operator {
    SAME_FUNCTION(FullConnectedGrad);
    DISABLE_COPY_AND_ASSIGN(FullConnectedGrad);
        INPUT_TAG(PRE_GRAD, OUT, DATA, WEIGHT, BIAS);
    };

    template <typename Context>
    Operator* CreateOp(FullConnectedGradParam &param, long *size);


    class FullConnectedGradOpProp : public OperatorProperty {
    public:
        FullConnectedGradOpProp();
        FullConnectedGradOpProp(const MatrixType &type);
        ~FullConnectedGradOpProp();
        virtual void InferShape(std::vector<Shape*> &inShape, Shape *outShape);
        virtual Operator* CreateOperator(Context context, std::vector<Blob*> &input, Blob* output,
                                         std::vector<Shape*> &inShape, Shape *outShape,
                                         std::map<std::string, Any> &args)  ;
        virtual void SwitchType(const MatrixType &type);
    private:
        FullConnectedGradParam* param;
    };
}

REGISTER_OP_PROPERTY(grad_fullConnected, FullConnectedGradOpProp);


#endif //MATRIX_FULLCONNECTEDGRADOP_H
