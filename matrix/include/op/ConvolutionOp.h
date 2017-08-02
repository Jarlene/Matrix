//
// Created by Jarlene on 2017/7/31.
//

#ifndef MATRIX_CONVOLUTIONOP_H
#define MATRIX_CONVOLUTIONOP_H

#include "Operator.h"

namespace matrix {

    class ConvolutionParam : Parameter {

    };

    template <class T, class Context>
    class ConvolutionOp : public Operator {
    SAME_FUNCTION(Convolution);
    DISABLE_COPY_AND_ASSIGN(Convolution);
    };

    template <typename Context>
    Operator* CreateOp(ConvolutionParam param, MatrixType type, std::vector<Shape> &in, std::vector<Shape> out);

}

#endif //MATRIX_CONVOLUTIONOP_H
