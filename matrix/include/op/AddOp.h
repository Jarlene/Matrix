//
// Created by Jarlene on 2017/7/25.
//

#ifndef MATRIX_ADDOP_H
#define MATRIX_ADDOP_H

#include "BaseOperator.h"

namespace matrix {

    class AddOp : public BaseOperator {

    };

}

INSTANTIATE_OPS(matrix::AddOp, add);
#endif //MATRIX_ADDOP_H
