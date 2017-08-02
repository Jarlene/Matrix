//
// Created by Jarlene on 2017/7/31.
//

#ifndef MATRIX_DROPOUTOP_H
#define MATRIX_DROPOUTOP_H

#include "Operator.h"

namespace matrix {

    class DropoutParam : Parameter{

    };

    template <class T, class Context>
    class DropoutOp : public Operator {
    public:
        explicit DropoutOp(DropoutParam &param);

        virtual bool Run() override ;

        virtual void AsyncRun() override ;

        virtual ~DropoutOp();

        virtual bool RunOnDevice() override ;

    DISABLE_COPY_AND_ASSIGN(DropoutOp);
    };

    template <typename Context>
    Operator* CreateOp(DropoutParam param, MatrixType type, std::vector<Shape> &in, std::vector<Shape> out);

}

#endif //MATRIX_DROPOUTOP_H
