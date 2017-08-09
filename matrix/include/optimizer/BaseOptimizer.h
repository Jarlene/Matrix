//
// Created by Jarlene on 2017/7/29.
//

#ifndef MATRIX_BASEOPTIMIZER_H
#define MATRIX_BASEOPTIMIZER_H

#include <unordered_map>

#include "matrix/include/base/Blob.h"

namespace matrix {

    template<int N>
    class BaseOptimizer {
    public:
        BaseOptimizer() = default;

        BaseOptimizer(const BaseOptimizer &) = default;

        BaseOptimizer(BaseOptimizer &&) = default;

        BaseOptimizer &operator=(const BaseOptimizer &) = default;

        BaseOptimizer &operator=(BaseOptimizer &&) = default;

        virtual ~BaseOptimizer() = default;

        virtual void Update(std::vector<Blob> dx, std::vector<Blob> x) = 0;

        virtual void Reset() {}

    protected:
//        std::unordered_map<>
    };

}


#endif //MATRIX_BASEOPTIMIZER_H
