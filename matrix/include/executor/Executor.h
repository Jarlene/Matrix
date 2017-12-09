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
        void train(const Symbol *symbol = nullptr);
        void syncTrain(const Symbol *symbol = nullptr);
        void update();
        void* evaluating(const Symbol *symbol = nullptr);

    private:
        void Init();
    private:
        BlockQueue<NodePtr> ready_;
        Graph* graph_{nullptr};
        std::mutex mutex_;
    };
}

#endif //MATRIX_EXECUTOR_H
