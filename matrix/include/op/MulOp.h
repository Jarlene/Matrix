//
// Created by Jarlene on 2017/7/25.
//

#ifndef MATRIX_MULOP_H
#define MATRIX_MULOP_H

#include "BaseOperator.h"

namespace matrix {

    class MulOp : public BaseOperator{

    };
}
INSTANTIATE_OPS(matrix::MulOp, mul);
#endif //MATRIX_MULOP_H
