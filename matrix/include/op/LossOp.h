//
// Created by Jarlene on 2017/7/31.
//

#ifndef MATRIX_LOSSOP_H
#define MATRIX_LOSSOP_H


#include "Operator.h"

namespace matrix {

    struct LossParam : public Parameter {
        LossParam(MatrixType matrixType);
    };

    template <class T, class Context>
    class LossOp : public Operator {
    SAME_FUNCTION(Loss);
    DISABLE_COPY_AND_ASSIGN(Loss);
    };

    template <typename Context>
    Operator* CreateOp(LossParam &param);


    class LossOpProp : public OperatorProperty {
    public:
        LossOpProp();
        LossOpProp(const MatrixType &type);
        ~LossOpProp();
        virtual void InferShape(std::vector<Shape> &inShape, std::vector<Shape> &outShape);
        virtual Operator* CreateOperator(Context context, std::vector<Blob> &input, std::vector<Blob> &output, std::vector<Shape> &inShape, std::vector<Shape> &outShape) ;
    private:
        LossParam* param;
    };
}

REGISTER_OP_PROPERTY(loss, LossOpProp);

#endif //MATRIX_LOSSOP_H
