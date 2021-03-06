//
// Created by Jarlene on 2017/7/29.
//

#ifndef MATRIX_ADAOPTIMIZER_H
#define MATRIX_ADAOPTIMIZER_H

#include "BaseOptimizer.h"

namespace matrix {

    class AdaGradOptimizer : public BaseOptimizer {
    public:
        AdaGradOptimizer(float learning_rate);

    };
}

#endif //MATRIX_ADAOPTIMIZER_H
