//
// Created by Jarlene on 2017/8/23.
//

#ifndef MATRIX_ADDGRADOP_H
#define MATRIX_ADDGRADOP_H


#include "Operator.h"

namespace matrix {

    struct AddGradParam : public Parameter {
        AddGradParam(MatrixType matrixType) : Parameter(matrixType) {
        }

    };

    template <class T, class xpu>
    class AddGradOp : public Operator {
    SAME_FUNCTION(AddGrad);
    DISABLE_COPY_AND_ASSIGN(AddGrad);
        INPUT_TAG(PRE_GRAD, OUT, INPUT1, INPUT2);
        OUTPUT_TAG(OUT_GRAD);
    };


    template <typename xpu>
    Operator* CreateOp(AddGradParam &param, long *size);


    class AddGradOpProp : public OperatorProperty {
    public:
        AddGradOpProp();
        AddGradOpProp(const MatrixType &type);
        ~AddGradOpProp();
        virtual void InferShape(std::vector<Shape*> &inShape, std::vector<Shape*> &outShape);
        virtual Operator* CreateOperator(Context context, std::vector<Blob*> &input, std::vector<Blob*> &output,
                                         std::vector<Shape*> &inShape, std::vector<Shape*> &outShape,
                                         std::map<std::string, Any> &args)  ;
        virtual void SwitchType(const MatrixType &type);
    private:
        AddGradParam* param;
    };

}

REGISTER_OP_PROPERTY(grad_add, AddGradOpProp);

#endif //MATRIX_ADDGRADOP_H
