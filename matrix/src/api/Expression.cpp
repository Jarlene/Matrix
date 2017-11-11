//
// Created by Jarlene on 2017/11/8.
//

#include "matrix/include/api/VariableSymbol.h"
#include "matrix/include/api/Expression.h"

namespace matrix {


    inline void convolution(ParameterCollection *parameterCollection,
                            const Symbol &input, const Shape &kernel, int filter_num,
                            Shape padding , Shape stride ,
                            Shape dilate , bool with_bias , int group) {



        Symbol weight = VariableSymbol::Create("kernel", kernel);
        parameterCollection->add_param(weight);
        Symbol convolution = Symbol("convolution")
                .SetInput("data", input)
                .SetInput("kernel", weight);
        if (with_bias) {
            Symbol bias = VariableSymbol::Create("bias");
            convolution.SetInput("bias", bias);
            parameterCollection->add_param(bias);
        }
        convolution.SetParam("padding", padding)
                .SetParam("stride", stride)
                .SetParam("dilate", dilate)
                .SetParam("group", group)
                .SetParam("filter_num", filter_num)
                .Build();

        parameterCollection->add(convolution);
    }


    inline void fullyConnected(ParameterCollection *parameterCollection,
                               const Symbol &input, int hiddenNum, bool with_bias) {
        Symbol weight = VariableSymbol::Create("kernel", ShapeN(0, hiddenNum));
        Symbol fullyConnected = Symbol("fullyConnected")
                .SetInput("data", input)
                .SetInput("weight", weight);
        parameterCollection->add_param(weight);
        if (with_bias) {
            Symbol bias = VariableSymbol::Create("bias");
            fullyConnected.SetInput("bias", bias);
            parameterCollection->add_param(bias);

        }
        fullyConnected.Build();
        parameterCollection->add(fullyConnected);
    }
}