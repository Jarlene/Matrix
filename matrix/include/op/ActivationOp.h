//
// Created by Jarlene on 2017/7/31.
//

#ifndef MATRIX_ACTIVATIONOP_H
#define MATRIX_ACTIVATIONOP_H

#include "Operator.h"

namespace matrix {

    struct ActivationParam : public Parameter {
        ActivationParam(MatrixType matrixType) : Parameter(matrixType) {

        }
    };

    template <class T, class Context>
    class ActivationOp : public Operator {
    SAME_FUNCTION(Activation);
    DISABLE_COPY_AND_ASSIGN(Activation);
    };


    template <typename Context>
    Operator* CreateOp(ActivationParam &param);

    class ActivationOpProp : public OperatorProperty {
    public:
        ActivationOpProp();
        ActivationOpProp(const MatrixType &type);
        ~ActivationOpProp();
        virtual void InferShape(std::vector<Shape> &inShape, std::vector<Shape> &outShape);
        virtual Operator* CreateOperator(Context context, std::vector<Blob> &input, std::vector<Blob> &output,
                                         std::vector<Shape> &inShape, std::vector<Shape> &outShape,
                                         std::map<std::string, Any> &args) ;
    private:
        ActivationParam* param;
    };

}

REGISTER_OP_PROPERTY(activation, ActivationOpProp);

#endif //MATRIX_ACTIVATIONOP_H
