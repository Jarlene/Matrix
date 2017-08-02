//
// Created by Jarlene on 2017/7/25.
//

#ifndef MATRIX_MULOP_H
#define MATRIX_MULOP_H

#include "Operator.h"

namespace matrix {

    class MulParam : public Parameter {

    };

    template <class T, class Context>
    class MulOp : public Operator {
    public:
        explicit MulOp(MulParam &param);

        virtual bool Run() override ;

        virtual void AsyncRun() override ;

        virtual ~MulOp();

        virtual bool RunOnDevice() override ;

    DISABLE_COPY_AND_ASSIGN(MulOp);
    };

    template <typename Context>
    Operator* CreateOp(MulParam param, MatrixType type, std::vector<Shape> &in, std::vector<Shape> out);

}
#endif //MATRIX_MULOP_H
