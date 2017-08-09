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
        Executor(const Symbol &symbol,  Context &context);
        std::vector<void*> runAsync();
        std::vector<void*> runSync();

    private:
        void Init();

    private:
        std::vector<NodePtr> fetches_;
        BlockQueue<NodePtr> ready_;
        BlockMap<NodePtr, int> depen_;
        Graph* graph_;
        std::mutex mutex_;
    };
}

#endif //MATRIX_EXECUTOR_H
