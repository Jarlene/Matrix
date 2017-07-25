//
// Created by Jarlene on 2017/7/25.
//

#ifndef MATRIX_DIVOP_H
#define MATRIX_DIVOP_H

#include "BaseOperator.h"

namespace matrix {

    class DivOp : public BaseOperator {

    };
}

INSTANTIATE_OPS(matrix::DivOp, div);
#endif //MATRIX_DIVOP_H
