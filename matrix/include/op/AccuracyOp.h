//
// Created by Jarlene on 2017/7/27.
//

#ifndef MATRIX_ACCURACYOP_H
#define MATRIX_ACCURACYOP_H

#include "Operator.h"


namespace matrix {

    class AccuracyParam {

    };

    template <class T, class Context>
    class AccuracyOp : public Operator {
    public:
        explicit AccuracyOp(AccuracyParam &param);

        virtual bool Run() override ;

        virtual void AsyncRun() override ;

        virtual ~AccuracyOp();

        virtual bool RunOnDevice() override ;

    DISABLE_COPY_AND_ASSIGN(AccuracyOp);
    };




    template <typename Context>
    Operator* CreateOp(AccuracyParam param, MatrixType type, std::vector<Shape> &in, std::vector<Shape> out);
}


#endif //MATRIX_ACCURACYOP_H
