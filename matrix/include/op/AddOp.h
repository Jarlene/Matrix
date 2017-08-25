//
// Created by Jarlene on 2017/7/25.
//

#ifndef MATRIX_ADDOP_H
#define MATRIX_ADDOP_H

#include "Operator.h"

namespace matrix {

    struct AddParam : public Parameter {
        AddParam(MatrixType matrixType) : Parameter(matrixType) {
        }

    };

    template <class T, class xpu>
    class AddOp : public Operator {
    SAME_FUNCTION(Add);
    DISABLE_COPY_AND_ASSIGN(Add);
        INPUT_TAG(INPUT1, INPUT2);
    };


    template <typename xpu>
    Operator* CreateOp(AddParam &param, long *size);


    class AddOpProp : public OperatorProperty {
    public:
        AddOpProp();
        AddOpProp(const MatrixType &type);
        ~AddOpProp();
        virtual void InferShape(std::vector<Shape*> &inShape, Shape *outShape);
        virtual Operator* CreateOperator(Context context, std::vector<Blob*> &input, Blob* output,
                                         std::vector<Shape*> &inShape, Shape *outShape,
                                         std::map<std::string, Any> &args)  ;
        virtual void SwitchType(const MatrixType &type);
    private:
        AddParam* param;
    };

}

REGISTER_OP_PROPERTY(add, AddOpProp);


#endif //MATRIX_ADDOP_H
