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
        INPUT_TAG(DATA, LABEL);
    };

    template <typename Context>
    Operator* CreateOp(LossParam &param, long *size);


    class LossOpProp : public OperatorProperty {
    public:
        LossOpProp();
        LossOpProp(const MatrixType &type);
        ~LossOpProp();
        virtual void InferShape(std::vector<Shape*> &inShape, Shape *outShape);
        virtual Operator* CreateOperator(Context context, std::vector<Blob*> &input, Blob* output,
                                         std::vector<Shape*> &inShape, Shape *outShape,
                                         std::map<std::string, Any> &args)  ;
        virtual void SwitchType(const MatrixType &type);
    private:
        LossParam* param;
    };
}

REGISTER_OP_PROPERTY(loss, LossOpProp);

#endif //MATRIX_LOSSOP_H
