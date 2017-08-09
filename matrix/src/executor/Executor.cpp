//
// Created by Jarlene on 2017/8/9.
//

#include "matrix/include/executor/Executor.h"

namespace matrix {

    Executor::Executor(const Symbol &symbol,  Context &context) {
        graph_ = new Graph(symbol,  context.phase == Phase::TRAIN);

    }

    std::vector<void *> Executor::runAsync() {
        return std::vector<void *>();
    }

    std::vector<void *> Executor::runSync() {
        return std::vector<void *>();
    }
}