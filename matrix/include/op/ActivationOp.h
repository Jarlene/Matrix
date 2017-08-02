//
// Created by Jarlene on 2017/7/31.
//

#ifndef MATRIX_ACTIVATIONOP_H
#define MATRIX_ACTIVATIONOP_H

#include "Operator.h"
#include "AccuracyOp.h"

namespace matrix {

    class ActivationParam : public Parameter {

    };

    template <class T, class Context>
    class ActivationOp : public Operator {
    SAME_FUNCTION(Activation);
    DISABLE_COPY_AND_ASSIGN(Activation);
    };


    template <typename Context>
    Operator* CreateOp(ActivationParam param, MatrixType type, std::vector<Shape> &in, std::vector<Shape> out);


}

#endif //MATRIX_ACTIVATIONOP_H
