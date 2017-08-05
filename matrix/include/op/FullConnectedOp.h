//
// Created by Jarlene on 2017/7/31.
//

#ifndef MATRIX_FULLCONNECTEDOP_H
#define MATRIX_FULLCONNECTEDOP_H

#include "Operator.h"

namespace matrix {


    struct  FullConnectedParam : public Parameter {
        FullConnectedParam(MatrixType matrixType);
    };

    template <class T, class Context>
    class FullConnectedOp : public Operator {
    SAME_FUNCTION(FullConnected);
    DISABLE_COPY_AND_ASSIGN(FullConnected);
    };

    template <typename Context>
    Operator* CreateOp(FullConnectedParam param, MatrixType type, std::vector<Shape> &in, std::vector<Shape> out);

}

#endif //MATRIX_FULLCONNECTEDOP_H
