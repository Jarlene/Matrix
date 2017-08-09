#include "matrix/include/op/ActivationOp.h"



namespace matrix {

    template <>
    Operator* CreateOp<GPU>(ActivationParam &param) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new ActivationOp<DType, GPU>(param);
        })
        return op;
    }


}