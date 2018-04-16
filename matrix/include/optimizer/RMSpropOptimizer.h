//
// Created by Jarlene on 2017/7/29.
//

#ifndef MATRIX_RMSPROPOPTIMIZER_H
#define MATRIX_RMSPROPOPTIMIZER_H

#include "BaseOptimizer.h"

namespace matrix {

    class RMSpropOptimizer : public BaseOptimizer {
    public:
        RMSpropOptimizer(float learning_rate);

    };
}

#endif //MATRIX_RMSPROPOPTIMIZER_H
