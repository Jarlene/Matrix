//
// Created by Jarlene on 2017/7/29.
//

#ifndef MATRIX_NESTEROPTIMIZER_H
#define MATRIX_NESTEROPTIMIZER_H

#include "BaseOptimizer.h"

namespace matrix {

    class NesterOptimizer : public BaseOptimizer {
    public:
        NesterOptimizer(float learning_rate);

    };
}

#endif //MATRIX_NESTEROPTIMIZER_H
