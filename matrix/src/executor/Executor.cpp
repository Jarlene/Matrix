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
        void *data = graph_->Evaluating(symbol);
        return data;
    }

    void Executor::update() {

        auto updateFunc = [this](NodePtr &node) {
            node->DirectRun();
        };
        ThreadPool pool(CPU_CORES + 1);
        for (auto it : graph_->GetUpdateNodes()) {
            pool.enqueue(updateFunc, it);
        }
    }

}