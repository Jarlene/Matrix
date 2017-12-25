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
        auto compute =[this](NodePtr &node) {
            node->Run();
        };
        ThreadPool pool(CPU_CORES + 1);
        for (auto node : graph_->GetGraphNodes()) {
            pool.enqueue(compute, node);
        }
        if (symbol != nullptr) {
            graph_->Accuracy(symbol)->Run();
        }
    }




    Executor::~Executor() {
        delete graph_;
    }


    void* Executor::evaluating(const Symbol *symbol) {
        if (symbol == nullptr) {
            return nullptr;
        }
        auto compute =[&](NodePtr &node) {
            node->Run();
        };
        ThreadPool pool(CPU_CORES + 1);
        for (auto node : graph_->GetForwardNodes()) {
            pool.enqueue(compute, node);
        }
        return graph_->Evaluating(symbol);
    }

    void Executor::update() {

        auto updateFunc = [this](NodePtr &node) {
            try {
                node->DirectRun();
            } catch (std::exception &e){
                Logger::Global()->Fatal("exception on node %s==> %s", node->ToString().c_str() , e.what());
            }
        };
        ThreadPool pool(CPU_CORES + 1);
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
            item->depenList.clear();
            item->depenList.insert(item->depenList.end(), item->inputs.begin(), item->inputs.end());
            item->depenList.sort();
            item->depenList.unique();
            int size = item->depenList.size();
            if (size == 0) {
                ready_.Put(item);
            }
        }
    }

    void Executor::syncTrain(const Symbol *symbol) {
        Init();

        auto compute =[&](NodePtr &node) {

            try {
                node->DirectRun();
            } catch (std::exception &e){
                Logger::Global()->Fatal("exception on node %d==> %s", node->ToString().c_str(), e.what());
            }

            {
                std::lock_guard<std::mutex> lock (mutex_);
                for (auto &item : node->outputs) {
                    if(graph_->GetNode(item.lock()->id_)) {
                        item.lock()->depenList.remove(node);
                        if (item.lock()->depenList.empty()) {
                            if (!ready_.Has(item.lock())) {
                                ready_.Put(item.lock());
                            }
                        }
                    }
                }
            }


        };

        ThreadPool pool(CPU_CORES + 1);
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
            graph_->Accuracy(symbol)->DirectRun();
        }
    }
}