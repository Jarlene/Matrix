//
// Created by Jarlene on 2017/7/31.
//

#ifndef MATRIX_DROPOUTOP_H
#define MATRIX_DROPOUTOP_H

#include "Operator.h"

namespace matrix {

    struct  DropoutParam : public Parameter{
        DropoutParam(MatrixType matrixType);
    };

    template <class T, class Context>
    class DropoutOp : public Operator {
    SAME_FUNCTION(Dropout);
    DISABLE_COPY_AND_ASSIGN(Dropout);
    };

    template <typename xpu>
    Operator* CreateOp(DropoutParam &param, long *size);


    class DropoutOpProp : public OperatorProperty {
    public:
        DropoutOpProp();
        DropoutOpProp(const MatrixType &type);
        ~DropoutOpProp();
        virtual void InferShape(std::vector<Shape*> &inShape, Shape* outShape);
        virtual Operator* CreateOperator(Context context, std::vector<Blob*> &input, Blob* output,
                                         std::vector<Shape*> &inShape, Shape* outShape,
                                         std::map<std::string, Any> &args)  ;
        virtual void SwitchType(const MatrixType &type);
    private:
        DropoutParam* param;
    };
}
REGISTER_OP_PROPERTY(dropout, DropoutOpProp);

#endif //MATRIX_DROPOUTOP_H
