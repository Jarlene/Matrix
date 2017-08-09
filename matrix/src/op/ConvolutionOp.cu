#include "matrix/include/op/ConvolutionOp.h"


namespace matrix {


    template <>
    Operator* CreateOp<GPU>(ConvolutionParam &param) {
        Operator *op = nullptr;
        TYPE_SWITCH(param.type, DType, {
            op = new ConvolutionOp<DType, GPU>(param);
        })
        return op;
    }
}