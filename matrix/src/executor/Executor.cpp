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
                node->SetData();
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
        std::vector<NodePtr> input;
        for(auto &item : graph_->GetGraphNodes()) {
            input.insert(input.end(), item->inputs.begin(), item->inputs.end());
            sort(input.begin(), input.end(), Graph::less);
            input.erase(unique(input.begin(), input.end()), input.end());
            int size = input.size();
            if (item->op == nullptr || size == 0) {
                ready_.Put(item);
            } else {
                if (!depen_.HasKey(item)) {
                    depen_.Put(item, size);
                }
            }
            input.clear();
        }
        ready_.Unique();
    }

    Executor::~Executor() {
        delete graph_;
    }
}