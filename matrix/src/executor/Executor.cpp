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

    void Executor::train(const Symbol *symbol) {
        Init();

        auto compute =[&](NodePtr &node) {

            if (node->op != nullptr && !node->isPlaceHolder && !node->isShared) {
                node->SetData();
                try {
                    node->op->AsyncRun();
                } catch (std::exception &e){
                    Logger::Global()->Fatal("exception on node %d==> %s", node->id_ , e.what());
                }
            }

            {
                std::lock_guard<std::mutex> lock (mutex_);
                for (auto &item : node->outputs) {
                    if(graph_->GetNode(item.lock()->id_)) {
                        item.lock()->depens_.remove(node);
                        if (item.lock()->depens_.empty()) {
                            if (!ready_.Has(item.lock())) {
                                ready_.Put(item.lock());
                            }
                        }
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
        int size = graph_->GetGraphNodes().size();
        while (true){
            auto node = ready_.Take();
            pool.enqueue(compute, node);
            size--;
            if (size == 0) {
                break;
            }
        }
        if (symbol != nullptr) {
            graph_->Accuracy(symbol)->SetData();
            graph_->Accuracy(symbol)->op->AsyncRun();
        }
        for (auto it : graph_->GetUpdateNodes()) {
            pool.enqueue(updateFunc, it);
        }
    }


    void Executor::Init() {
        for(auto &item : graph_->GetGraphNodes()) {
            if (item->op == nullptr || item->isShared) {
                ready_.Put(item);
                continue;
            }
            item->depens_.clear();
            item->depens_.insert(item->depens_.end(), item->inputs.begin(), item->inputs.end());
            item->depens_.sort();
            item->depens_.unique();
            int size = item->depens_.size();
            if (size == 0) {
                ready_.Put(item);
            }
        }
    }

    Executor::~Executor() {
        delete graph_;
    }


    void Executor::evaluating() {

    }
}