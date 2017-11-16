//
// Created by Jarlene on 2017/8/9.
//

#ifndef MATRIX_EXECUTOR_H
#define MATRIX_EXECUTOR_H

#include "matrix/include/utils/BlockQueue.h"
#include "matrix/include/utils/BlockMap.h"
#include "matrix/include/base/Graph.h"
#include "matrix/include/api/Symbol.h"
#include "matrix/include/scheduler/ThreadPool.h"
#include "matrix/include/optimizer/BaseOptimizer.h"

namespace matrix {


    class Executor {
    public:
        Executor(const Symbol &symbol,  Context &context, BaseOptimizer *optimizer);
        ~Executor();
        void runAsync();
        void runSync();

    private:
        void Init();

    private:
        BlockQueue<NodePtr> ready_;
        Graph* graph_{nullptr};
        std::mutex mutex_;
    };
}

#endif //MATRIX_EXECUTOR_H
