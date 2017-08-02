//
// Created by Jarlene on 2017/7/25.
//

#ifndef MATRIX_DIVOP_H
#define MATRIX_DIVOP_H

#include "Operator.h"

namespace matrix {

    class DivParam : Parameter {

    };

    template <class T, class Context>
    class DivOp : public Operator {
    public:
        explicit DivOp(DivParam &param);

        virtual bool Run() override ;

        virtual void AsyncRun() override ;

        virtual ~DivOp();

        virtual bool RunOnDevice() override ;

    DISABLE_COPY_AND_ASSIGN(DivOp);
    };


    template <typename Context>
    Operator* CreateOp(DivParam param, MatrixType type, std::vector<Shape> &in, std::vector<Shape> out);

}

#endif //MATRIX_DIVOP_H
