//
// Created by Jarlene on 2017/7/29.
//

#ifndef MATRIX_ADAMOPTIMIZER_H
#define MATRIX_ADAMOPTIMIZER_H

#include "BaseOptimizer.h"

namespace matrix {

    class AdamOptimizer : public BaseOptimizer {
    public:
        AdamOptimizer(float learning_rate);

    };
}

#endif //MATRIX_ADAMOPTIMIZER_H
