//
// Created by Jarlene on 2017/8/9.
//

#ifndef MATRIX_PLACEHOLDER_H
#define MATRIX_PLACEHOLDER_H

#include "Operator.h"

namespace matrix {

    struct PlaceHolderParam : public Parameter {
        PlaceHolderParam(MatrixType matrixType) : Parameter(matrixType) {

        }

    };

    template <class T, class xpu>
    class PlaceHolderOp : public Operator {
        SAME_FUNCTION(PlaceHolder);
        DISABLE_COPY_AND_ASSIGN(PlaceHolder);
        INPUT_TAG(INPUT);
        OUTPUT_TAG(OUT);
    };



    template <typename xpu>
    Operator* CreateOp(PlaceHolderParam &param);


    class PlaceHolderOpProp : public OperatorProperty {
    public:
        PlaceHolderOpProp();
        PlaceHolderOpProp(const MatrixType &type);
        ~PlaceHolderOpProp();
        virtual void InferShape(std::vector<Shape> &inShape, std::vector<Shape> &outShape);
        virtual Operator* CreateOperator(Context context, std::vector<Blob> &input, std::vector<Blob> &output,
                                         std::vector<Shape> &inShape, std::vector<Shape> &outShape,
                                         std::map<std::string, Any> &args) ;
    private:
        PlaceHolderParam* param;
    };
}

REGISTER_OP_PROPERTY(placeHolder, PlaceHolderOpProp);


#endif //MATRIX_PLACEHOLDER_H
