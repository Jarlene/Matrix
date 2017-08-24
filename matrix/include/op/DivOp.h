//
// Created by Jarlene on 2017/7/25.
//

#ifndef MATRIX_DIVOP_H
#define MATRIX_DIVOP_H

#include "Operator.h"

namespace matrix {

    struct  DivParam : public Parameter {
        DivParam(MatrixType matrixType);
    };

    template <class T, class Context>
    class DivOp : public Operator {
    SAME_FUNCTION(Div);
    DISABLE_COPY_AND_ASSIGN(Div);
    };


    template <typename xpu>
    Operator* CreateOp(DivParam &param, long *size);


    class DivOpProp : public OperatorProperty {
    public:
        DivOpProp();
        DivOpProp(const MatrixType &type);
        ~DivOpProp();
        virtual void InferShape(std::vector<Shape*> &inShape, std::vector<Shape*> &outShape);
        virtual Operator* CreateOperator(Context context, std::vector<Blob*> &input, std::vector<Blob*> &output,
                                         std::vector<Shape*> &inShape, std::vector<Shape*> &outShape,
                                         std::map<std::string, Any> &args)  ;
        virtual void SwitchType(const MatrixType &type);
    private:
        DivParam* param;
    };

}
REGISTER_OP_PROPERTY(div, DivOpProp);

#endif //MATRIX_DIVOP_H
