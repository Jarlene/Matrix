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

    template <class T, class Context>
    class AddOp : public Operator {
    SAME_FUNCTION(Add);
    DISABLE_COPY_AND_ASSIGN(Add);
        INPUT_TAG(INPUT1, INPUT2);
        OUTPUT_TAG(OUT);
    private:
        Context context;
        Shape inShape;
        Shape outShape;
    };


    template <typename Context>
    Operator* CreateOp(AddParam &param);


    class AddOpProp : public OperatorProperty {
    public:
        virtual void InferShape(std::vector<Shape> *inShape, std::vector<Shape> *outShape) const;
        virtual Operator* CreateOperator(std::vector<Shape> *inShape, std::vector<Shape> *outShape) const ;
    private:
        AddParam param;
    };

}

#endif //MATRIX_ADDOP_H
