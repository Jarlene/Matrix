//
// Created by Jarlene on 2017/7/31.
//

#ifndef MATRIX_POOLOP_H
#define MATRIX_POOLOP_H

#include "Operator.h"

namespace matrix {

    class PoolingParam : Parameter {

    };

    template <class T, class Context>
    class PoolingOp : public Operator {
    public:
        explicit PoolingOp(PoolingParam &param);

        virtual bool Run() override ;

        virtual void AsyncRun() override ;

        virtual ~PoolingOp();

        virtual bool RunOnDevice() override ;

    DISABLE_COPY_AND_ASSIGN(PoolingOp);
    };

    template <typename Context>
    Operator* CreateOp(PoolingParam param, MatrixType type, std::vector<Shape> &in, std::vector<Shape> out);

}
#endif //MATRIX_POOLOP_H
