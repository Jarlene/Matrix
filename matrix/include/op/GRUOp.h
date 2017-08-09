//
// Created by Jarlene on 2017/7/28.
//

#ifndef MATRIX_GRUOP_H
#define MATRIX_GRUOP_H

#include "Operator.h"

namespace matrix {

    struct  GRUParam : public Parameter {
        GRUParam(MatrixType matrixType);
    };

    template <class T, class Context>
    class GRUOp : public Operator {
    SAME_FUNCTION(GRU);
    DISABLE_COPY_AND_ASSIGN(GRU);
    };

    template <typename Context>
    Operator* CreateOp(GRUParam &param, long *size);


    class GRUOpProp : public OperatorProperty {
    public:
        GRUOpProp();
        GRUOpProp(const MatrixType &type);
        ~GRUOpProp();
        virtual void InferShape(std::vector<Shape> &inShape, std::vector<Shape> &outShape);
        virtual Operator* CreateOperator(Context context, std::vector<Blob> &input, std::vector<Blob> &output,
                                         std::vector<Shape> &inShape, std::vector<Shape> &outShape,
                                         std::map<std::string, Any> &args) ;
    private:
        GRUParam* param;
    };
}
REGISTER_OP_PROPERTY(gru, GRUOpProp);
#endif //MATRIX_GRUOP_H
