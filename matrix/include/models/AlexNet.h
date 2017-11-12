//
// Created by Jarlene on 2017/7/31.
//

#ifndef MATRIX_ALEXNET_H
#define MATRIX_ALEXNET_H

#include "../api/Symbol.h"
#include "../api/VariableSymbol.h"
#include "../base/Shape.h"
#include "../api/MatrixType.h"

namespace matrix {


    Symbol AlexSymbol(Symbol &input, Symbol &label) {


        auto con1 = Symbol("convolution")
                .SetInput("x", input)
                .SetParam("filter", ShapeN(11, 11))
                .SetParam("padding", ShapeN(0,0))
                .SetParam("stride", ShapeN(4,4))
                .SetParam("dilate", ShapeN(1,1))
                .SetParam("filter_num", 96)
                .SetParam("with_bias", true)
                .SetParam("group", 1)
                .Build();

        auto act1 = Symbol("activation")
                .SetInput("data", con1)
                .SetParam("type", kRelu)
                .Build();

        auto pool1 = Symbol("pooling")
                .SetInput("act1", act1)
                .SetParam("filter", ShapeN(3,3))
                .SetParam("stride", ShapeN(2,2))
                .SetParam("type", kMax)
                .Build();

        auto con2 = Symbol("convolution")
                .SetInput("data", pool1)
                .SetParam("filter", ShapeN(5, 5))
                .SetParam("padding", ShapeN(2,2))
                .SetParam("stride", ShapeN(1,1))
                .SetParam("dilate", ShapeN(1,1))
                .SetParam("filter_num", 256)
                .SetParam("with_bias", true)
                .SetParam("group", 1)
                .Build();

        auto act2 = Symbol("activation")
                .SetInput("data", con2)
                .SetParam("type", kRelu)
                .Build();

        auto pool2 = Symbol("pooling")
                .SetInput("act1", act2)
                .SetParam("filter", ShapeN(3,3))
                .SetParam("stride", ShapeN(2,2))
                .SetParam("type", kMax)
                .Build();

        auto con3 = Symbol("convolution")
                .SetInput("data", pool2)
                .SetParam("filter", ShapeN(3, 3))
                .SetParam("padding", ShapeN(1,1))
                .SetParam("stride", ShapeN(1,1))
                .SetParam("dilate", ShapeN(1,1))
                .SetParam("filter_num", 384)
                .SetParam("with_bias", true)
                .SetParam("group", 1)
                .Build();

        auto act3 = Symbol("activation")
                .SetInput("data", con3)
                .SetParam("type", kRelu);

        auto con4 = Symbol("convolution")
                .SetInput("data", act3)
                .SetParam("filter", ShapeN(3, 3))
                .SetParam("padding", ShapeN(1,1))
                .SetParam("stride", ShapeN(1,1))
                .SetParam("dilate", ShapeN(1,1))
                .SetParam("filter_num", 384)
                .SetParam("with_bias", true)
                .SetParam("group", 1)
                .Build();

        auto act4 = Symbol("activation")
                .SetInput("data", con4)
                .SetParam("type", kRelu)
                .Build();

        auto con5 = Symbol("convolution")
                .SetInput("data", act4)
                .SetParam("filter", ShapeN(3, 3))
                .SetParam("padding", ShapeN(1,1))
                .SetParam("stride", ShapeN(1,1))
                .SetParam("dilate", ShapeN(1,1))
                .SetParam("filter_num", 256)
                .SetParam("group", 1)
                .SetParam("with_bias", true)
                .Build();

        auto act5 = Symbol("activation")
                .SetInput("data", con5)
                .SetParam("type", kRelu)
                .Build();

        auto pool5 = Symbol("pooling")
                .SetInput("act1", act2)
                .SetParam("filter", ShapeN(3,3))
                .SetParam("stride", ShapeN(2,2))
                .SetParam("type", kMax)
                .Build();

        auto fc1 = Symbol("fullyconnected")
                .SetInput("data", pool5)
                .SetParam("hide_num",6)
                .SetParam("with_bias", true)
                .Build();

        auto act6 = Symbol("activation")
                .SetInput("data", fc1)
                .SetParam("type", kRelu)
                .Build();

        auto drop = Symbol("dropout")
                .SetInput("data", act6)
                .SetParam("ratio", 0.5f)
                .Build();

        auto fc2 = Symbol("fullyconnected")
                .SetInput("data", drop)
                .SetParam("hide_num",1)
                .SetParam("with_bias", true)
                .Build();

        auto act7 = Symbol("activation")
                .SetInput("data", fc2)
                .SetParam("type", kRelu)
                .Build();

        auto drop2 = Symbol("dropout")
                .SetInput("data", act7)
                .SetParam("ratio", 0.5f)
                .Build();

        auto fc3 = Symbol("fullyconnected")
                .SetInput("data", drop2)
                .SetParam("hide_num", 1)
                .SetParam("with_bias", true)
                .Build();

        auto softmax = Symbol("output")
                .SetInput("data", fc3)
                .SetParam("type", kSoftmax)
                .Build();
        return softmax;
    }

}





#endif //MATRIX_ALEXNET_H
