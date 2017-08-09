//
// Created by Jarlene on 2017/7/31.
//

#ifndef MATRIX_POOLOP_H
#define MATRIX_POOLOP_H

#include "Operator.h"

namespace matrix {

    struct PoolingParam : public Parameter {
        PoolingParam(MatrixType matrixType);
    };

    template <class T, class Context>
    class PoolingOp : public Operator {
    SAME_FUNCTION(Pooling);
    DISABLE_COPY_AND_ASSIGN(Pooling);
    };

    template <typename xpu>
    Operator* CreateOp(PoolingParam &param);


    class PoolingOpProp : public OperatorProperty {
    public:
        PoolingOpProp();
        PoolingOpProp(const MatrixType &type);
        ~PoolingOpProp();
        virtual void InferShape(std::vector<Shape> &inShape, std::vector<Shape> &outShape);
        virtual Operator* CreateOperator(Context context, std::vector<Blob> &input, std::vector<Blob> &output, std::vector<Shape> &inShape, std::vector<Shape> &outShape) ;
    private:
        PoolingParam* param;
    };
}

REGISTER_OP_PROPERTY(pooling, PoolingOpProp);

#endif //MATRIX_POOLOP_H
