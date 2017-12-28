

#include "matrix/inlcude/utils/Cuda.h"
#include "matrix/include/op/AccuracyOp.h"


namespace matrix {

    template <>
    Operator* CreateOp<GPU>(AccuracyParam &param) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new AccuracyOp<DType, GPU>(param);
        })
        return op;
    }

}