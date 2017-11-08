//
// Created by Jarlene on 2017/11/7.
//

#include "matrix/include/optimizer/SGDOptimizer.h"

namespace matrix {


    void SGDOptimizer::Update(std::vector<Blob> dx, std::vector<Blob> x) {
        int size = dx.size();
        for (int i = 0; i < size; ++i) {
            Blob di = dx.at(i);
            Blob xi = x.at(i);
        }

    }

    SGDOptimizer::SGDOptimizer(float learning_rate) : learning_rate(learning_rate){

    }
}