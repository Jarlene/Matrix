//
// Created by Jarlene on 2017/8/9.
//

#ifndef MATRIX_EXECUTOR_H
#define MATRIX_EXECUTOR_H

#include "matrix/include/base/Graph.h"
#include "matrix/include/api/Symbol.h"
#include "matrix/include/scheduler/ThreadPool.h"
#include "matrix/include/optimizer/BaseOptimizer.h"

namespace matrix {


    class Executor {
    public:
        Executor(const Symbol &symbol,  Context &context);
        std::vector<void*> runAsync();
        std::vector<void*> runSync();

    private:
        void Init();

    private:
        Graph* graph_;
    };
}

#endif //MATRIX_EXECUTOR_H
