//
// Created by Jarlene on 2017/11/6.
//

#ifndef MATRIX_POOLINGGRADOP_H
#define MATRIX_POOLINGGRADOP_H

#include "Operator.h"

namespace matrix {


    struct PoolingGradParam : public Parameter {
        PoolingGradParam(MatrixType matrixType) : Parameter(matrixType) {

        };
    };

    template <class T, class Context>
    class PoolingGradOp : public Operator {
    SAME_FUNCTION(PoolingGrad);
    DISABLE_COPY_AND_ASSIGN(PoolingGrad)
        INPUT_TAG(PRE_GRAG, SELF_OUT, INPUT);
    };


    template <typename xpu>
    Operator* CreateOp(PoolingGradParam &param, long *size);

    class PoolingGradOpProp : public OperatorProperty {
    public:
        PoolingGradOpProp();
        PoolingGradOpProp(const MatrixType &type);
        ~PoolingGradOpProp();
        virtual void InferShape(std::vector<Shape*> &inShape, Shape* outShape);
        virtual Operator* CreateOperator(Context context, std::vector<Blob*> &input, Blob* output,
                                         std::vector<Shape*> &inShape, Shape* outShape,
                                         std::map<std::string, Any> &args)  ;
        virtual void SwitchType(const MatrixType &type);
    private:
        PoolingGradParam* param;
    };
}



REGISTER_OP_PROPERTY(grad_pooling, PoolingGradOpProp);



#endif //MATRIX_POOLINGGRADOP_H
