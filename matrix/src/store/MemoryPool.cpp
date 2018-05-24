//
// Created by Jarlene on 2017/3/20.
//

#include "matrix/include/store/MemoryPool.h"

namespace matrix {

    MemoryPool::MemoryPool(Allocator *allocator,
                           size_t sizeLimit,
                           const std::string &name) : allocator_(allocator),
                                                      sizeLimit_(sizeLimit),
                                                      poolMemorySize_(0),
                                                      name_(name)
                                                      {

    }

    void *MemoryPool::dynamicAllocate(size_t size) {
        if (size <= 0) {
            return nullptr;
        }
        if (sizeLimit_ > 0) {
            std::lock_guard<std::mutex> guard(mutex_);
            auto it = pool_.find(size);
            if (it == pool_.end() || it->second.size() == 0) {
                if (poolMemorySize_ >= sizeLimit_) {
                    freeAll();
                }
                return allocator_->alloc(size);
            } else {
                auto buf = it->second.back();
                it->second.pop_back();
                poolMemorySize_ -= size;
                return buf;
            }
        } else {
            poolMemorySize_ += size;
            auto d = allocator_->alloc(size);
            pools.push_back(d);
            return d;
        }
    }

    void MemoryPool::staticAllocate(const std::map<int, size_t> &colorSize) {
        {
            std::lock_guard<std::mutex> guard(mutex_);
            for (auto & item : colorSize) {
                if (item.second > 0) {
                    void* data = allocator_->alloc(item.second);
                    poolMemorySize_ += item.second;
                    memroy_[item.first] = data;
                }
            }
        }
    }

    void MemoryPool::freeMemory(int color) {
        {
            std::lock_guard<std::mutex> guard(mutex_);
            if (memroy_.count(color) > 0) {
                allocator_->free(memroy_[color]);
                memroy_.erase(color);
            }
        }
    }

    void MemoryPool::freeMemory(void *ptr, size_t n) {
        allocator_->free(ptr);
        poolMemorySize_ -= n;
    }

    void *MemoryPool::getMemory(int color) {
        if (memroy_.count(color) > 0) {
            return memroy_[color];
        }
        return nullptr;
    }

    void MemoryPool::freeAll() {
        for (auto it : pool_) {
            for (auto ptr : it.second) {
                allocator_->free(ptr);
            }
        }
        poolMemorySize_ = 0;
        pool_.clear();

        for (auto &it : memroy_) {
            allocator_->free(it.second);
        }
        memroy_.clear();
        for(auto &s : pools) {
            allocator_->free(s);
        }
        pools.clear();
    }

    void MemoryPool::PrintMemory() {
        MLOG(INFO) << "the all memory is " << (poolMemorySize_ * 1.0f) / (1024 * 1024) << " MB";
    }
}
