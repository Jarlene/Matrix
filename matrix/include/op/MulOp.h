//
// Created by Jarlene on 2017/7/25.
//

#ifndef MATRIX_MULOP_H
#define MATRIX_MULOP_H

#include "Operator.h"

namespace matrix {

    struct MulParam : public Parameter {
        MulParam(MatrixType matrixType) : Parameter(matrixType) {

        }
    };

    template <class T, class Context>
    class MulOp : public Operator {
    SAME_FUNCTION(Mul);
    DISABLE_COPY_AND_ASSIGN(Mul);
        INPUT_TAG(INPUT1, INPUT2);
        OUTPUT_TAG(OUT);
    };

    template <typename Context>
    Operator* CreateOp(MulParam &param);


    class MulOpProp : public OperatorProperty {
    public:
        MulOpProp();
        MulOpProp(const MatrixType type);
        ~MulOpProp();
        virtual void InferShape(std::vector<Shape> &inShape, std::vector<Shape> &outShape);
        virtual Operator* CreateOperator(Context context, std::vector<Blob> &input, std::vector<Blob> &output, std::vector<Shape> &inShape, std::vector<Shape> &outShape) ;
    private:
        MulParam* param;
    };
}

REGISTER_OP_PROPERTY(mul, MulOpProp);
#endif //MATRIX_MULOP_H
