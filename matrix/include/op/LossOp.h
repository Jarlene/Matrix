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
    SAME_FUNCTION(Loss);
    DISABLE_COPY_AND_ASSIGN(Loss);
    };

    template <typename Context>
    Operator* CreateOp(LossParam param, MatrixType type, std::vector<Shape> &in, std::vector<Shape> out);

}

#endif //MATRIX_LOSSOP_H
