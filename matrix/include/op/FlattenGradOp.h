//
// Created by Jarlene on 2017/11/9.
//

#ifndef MATRIX_FLATTENGRADOP_H
#define MATRIX_FLATTENGRADOP_H


#include "Operator.h"
namespace matrix {


    struct FlattenGradParam : public Parameter{
        FlattenGradParam(MatrixType matrixType): Parameter(matrixType) {

        };
    };

    template <class T, class Context>
    class FlattenGradOp : public Operator {
    SAME_FUNCTION(FlattenGrad);
    DISABLE_COPY_AND_ASSIGN(FlattenGrad);
    };

    template <typename xpu>
    Operator* CreateOp(FlattenGradParam &param, long *size);


    class FlattenGradOpProp : public OperatorProperty {
    public:
        FlattenGradOpProp();
        FlattenGradOpProp(const MatrixType &type);
        ~FlattenGradOpProp();
        virtual void InferShape(std::vector<Shape*> &inShape, Shape* outShape);
        virtual Operator* CreateOperator(Context context, std::vector<Blob*> &input, Blob* output,
                                         std::vector<Shape*> &inShape, Shape* outShape,
                                         std::map<std::string, Any> &args)  ;
        virtual void SwitchType(const MatrixType &type);
    private:
        FlattenGradParam* param;
    };

}

REGISTER_OP_PROPERTY(grad_flatten, FlattenGradOpProp);


#endif //MATRIX_FLATTENGRADOP_H
