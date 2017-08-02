//
// Created by Jarlene on 2017/7/31.
//

#ifndef MATRIX_LOSSOP_H
#define MATRIX_LOSSOP_H


#include "Operator.h"

namespace matrix {

    class LossParam : Parameter {

    };

    template <class T, class Context>
    class LossOp : public Operator {
    public:
        explicit LossOp(LossParam &param);

        virtual bool Run() override ;

        virtual void AsyncRun() override ;

        virtual ~LossOp();

        virtual bool RunOnDevice() override ;

    DISABLE_COPY_AND_ASSIGN(LossOp);
    };

    template <typename Context>
    Operator* CreateOp(LossParam param, MatrixType type, std::vector<Shape> &in, std::vector<Shape> out);

}

#endif //MATRIX_LOSSOP_H
