//
// Created by Jarlene on 2017/8/7.
//

#include "matrix/include/op/AddOp.h"

namespace matrix {



    template <>
    Operator* CreateOp<GPU>(AddParam &param) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new AddOp<DType, GPU>(param);
        })
        return op;
    }
}









