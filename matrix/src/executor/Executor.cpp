//
// Created by Jarlene on 2017/8/9.
//
#include <exception>
#include "matrix/include/executor/Executor.h"

namespace matrix {

    Executor::Executor(const Symbol &symbol,  Context &context, BaseOptimizer *optimizer) {
        graph_ = new Graph(symbol, optimizer, context.phase == Phase::TRAIN);
        graph_->Optimize();
        graph_->AllocateGraph();
    }

    void Executor::runAsync() {
        Init();

        auto compute =[&](NodePtr &node) {
            if (node->op != nullptr && !node->isPlaceHolder) {
                node->SetData();
                try {
                    node->op->AsyncRun();
                } catch (std::exception &e){
                    Logger::Global()->Fatal("exception on node %d==> %s", node->id_ , e.what());
                }
            }
            {
                std::unique_lock<std::mutex> lock (this->mutex_);
                for (auto &item : node->outputs) {
                    item.lock()->depens_.remove(node);
                    if (item.lock()->depens_.empty()) {
                        ready_.Put(item.lock());
                    }
                }
            }

        };

        auto updateFunc = [&](NodePtr &node) {
            node->SetData();
            try {
                node->op->AsyncRun();
            } catch (std::exception &e){
                Logger::Global()->Fatal("exception on node %d==> %s", node->id_ , e.what());
            }
        };

        ThreadPool pool(CPU_CORES);
        while (ready_.Size() != 0){
            auto node = ready_.Take();
            pool.enqueue(compute, node);
        }
        for (auto it : graph_->GetUpdateNodes()) {
            pool.enqueue(updateFunc, it);
        }
    }

    void Executor::runSync() {
        for (auto &node : graph_->GetGraphNodes()) {
            if (node->op != nullptr) {
                node->SetData();
                try {
                    node->op->AsyncRun();
                } catch (std::exception &e){
                    Logger::Global()->Fatal("exception on node %d==> %s", node->id_ , e.what());
                }
            }
        }
        for (auto &it : graph_->GetUpdateNodes()) {
            if (it->op != nullptr) {
                it->SetData();
                try {
                    it->op->AsyncRun();
                } catch (std::exception &e){
                    Logger::Global()->Fatal("exception on node %d==> %s", it->id_ , e.what());
                }
            }
        }
    }

    void Executor::Init() {
        for(auto &item : graph_->GetGraphNodes()) {
            item->depens_.clear();
            item->depens_.insert(item->depens_.end(), item->inputs.begin(), item->inputs.end());
            item->depens_.sort();
            item->depens_.unique();
            int size = item->depens_.size();
            if (item->op == nullptr || size == 0) {
                ready_.Put(item);
            }
        }
    }

    Executor::~Executor() {
        delete graph_;
    }
}