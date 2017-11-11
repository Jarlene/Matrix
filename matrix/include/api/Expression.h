//
// Created by Jarlene on 2017/11/8.
//

#ifndef MATRIX_EXPRESSION_H
#define MATRIX_EXPRESSION_H


#include "Symbol.h"
#include "ParameterCollection.h"

namespace matrix {







    inline void convolution(ParameterCollection *parameterCollection,
                            const Symbol &input, const Shape &kernel, int output_channel,
                            Shape padding = ShapeN(0, 0), Shape stride = ShapeN(1, 1),
                            Shape dilate = ShapeN(1, 1), bool bias = true, int group = 1);


    inline void fullyConnected(ParameterCollection *parameterCollection,
                               const Symbol &input, int hiddenNum, bool bias = true);


}


#endif //MATRIX_EXPRESSION_H
