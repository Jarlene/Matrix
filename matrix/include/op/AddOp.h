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

        Shape inShape;
        Shape outShape;
        std::vector<Blob> in;
        Blob* out;
    };

    template <class T, class xpu>
    class AddOp : public Operator {
    SAME_FUNCTION(Add);
    DISABLE_COPY_AND_ASSIGN(Add);
        INPUT_TAG(INPUT1, INPUT2);
        OUTPUT_TAG(OUT);
    private:
        Shape inShape;
        Shape outShape;
    };


    template <typename xpu>
    Operator* CreateOp(AddParam &param);


    class AddOpProp : public OperatorProperty {
    public:
        AddOpProp();
        AddOpProp(const MatrixType type);
        ~AddOpProp();
        virtual void InferShape(std::vector<Shape> &inShape, std::vector<Shape> &outShape);
        virtual Operator* CreateOperator(Context context, std::vector<Blob> &input, std::vector<Blob> &output, std::vector<Shape> &inShape, std::vector<Shape> &outShape) ;
    private:
        AddParam* param;
    };

}

REGISTER_OP_PROPERTY(add, AddOpProp);


#endif //MATRIX_ADDOP_H
