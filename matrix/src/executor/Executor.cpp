//
// Created by Jarlene on 2017/8/9.
//

#include "matrix/include/executor/Executor.h"

namespace matrix {

    Executor::Executor(const Symbol &symbol,  Context &context, BaseOptimizer *optimizer) {
        graph_ = new Graph(symbol, optimizer, context.phase == Phase::TRAIN);
        graph_->Optimize();
        graph_->AllocateGraph(fetches_);
    }

    std::vector<void *> Executor::runAsync() {
        Init();

        auto callback = [&](const NodePtr node) {
            for (auto &item : node->outputs) {

                int val = depen_.Take(item.lock());
                if (val <= 1) {
                    this->ready_.Put(item.lock());
                } else {
                    depen_.Put(item.lock(), val - 1);
                }
            }
        };
        auto compute =[&](NodePtr node) {
            {
                std::unique_lock<std::mutex> lock (mutex_);
                if (node->op != nullptr && !node->isPlaceHolder) {
                    node->SetData();
                    node->op->AsyncRun();
                }
                callback(node);
            }

        };

        ThreadPool pool(CPU_CORES);
        while (depen_.Size() != 0 || ready_.Size() != 0){
            auto node = ready_.Take();
//            compute(node);
            pool.enqueue(compute, node);
        }

        std::vector<void*> result;
        for(auto &it: fetches_) {
            result.push_back(it->data_);
        }
        return result;
    }

    std::vector<void *> Executor::runSync() {
        for (auto &node : graph_->GetGraphNodes()) {
            if (node->op != nullptr) {
                node->op->AsyncRun();
            }
        }
        std::vector<void*> result;
        for(auto &it: fetches_) {
            result.push_back(it->data_);
        }
        return result;
    }

    void Executor::Init() {
        depen_.Clear();
        ready_.Clear();
        for(auto &item : graph_->GetGraphNodes()) {
            int size = item->inputs.size();
            if (item->op == nullptr || size == 0) {
                ready_.Put(item);
            } else {
                if (!depen_.HasKey(item)) {
                    depen_.Put(item, size);
                }
            }
        }
    }

    Executor::~Executor() {
        delete graph_;
    }
}