//
// Created by Jarlene on 2017/7/28.
//

#ifndef MATRIX_GRUOP_H
#define MATRIX_GRUOP_H

#include "Operator.h"

namespace matrix {

    class GRUParam : Parameter {

    };

    template <class T, class Context>
    class GRUOp : public Operator {
    public:
        explicit GRUOp(GRUParam &param);

        virtual bool Run() override ;

        virtual void AsyncRun() override ;

        virtual ~GRUOp();

        virtual bool RunOnDevice() override ;

    DISABLE_COPY_AND_ASSIGN(GRUOp);
    };

    template <typename Context>
    Operator* CreateOp(GRUParam param, MatrixType type, std::vector<Shape> &in, std::vector<Shape> out);

}

#endif //MATRIX_GRUOP_H
