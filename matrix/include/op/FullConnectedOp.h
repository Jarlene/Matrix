//
// Created by Jarlene on 2017/7/31.
//

#ifndef MATRIX_FULLCONNECTEDOP_H
#define MATRIX_FULLCONNECTEDOP_H

#include "Operator.h"

namespace matrix {


    class FullConnectedParam : Parameter {

    };

    template <class T, class Context>
    class FullConnectedOp : public Operator {
    public:
        explicit FullConnectedOp(FullConnectedParam &param);

        virtual bool Run() override ;

        virtual void AsyncRun() override ;

        virtual ~FullConnectedOp();

        virtual bool RunOnDevice() override ;

    DISABLE_COPY_AND_ASSIGN(FullConnectedOp);
    };

    template <typename Context>
    Operator* CreateOp(FullConnectedParam param, MatrixType type, std::vector<Shape> &in, std::vector<Shape> out);

}

#endif //MATRIX_FULLCONNECTEDOP_H
