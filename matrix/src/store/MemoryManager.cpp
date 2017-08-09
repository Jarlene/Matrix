//
// Created by Jarlene on 2017/3/20.
//

#include "matrix/include/store/GpuAllocator.h"
#include "matrix/include/store/CpuAllocator.h"
#include "matrix/include/store/MemoryManager.h"

namespace matrix {


    MemoryPool *MemoryManager::GetGpuMemoryPool(int deviceId) {
        {
            std::lock_guard<std::mutex> guard(mutex_);
            if (deviceId < static_cast<int>(gpuAllocator_.size()) &&
                (gpuAllocator_[deviceId] != nullptr)) {
                return gpuAllocator_[deviceId];
            }
        }

        {
            std::lock_guard<std::mutex> guard(mutex_);
            if (deviceId >= static_cast<int>(gpuAllocator_.size())) {
                gpuAllocator_.resize(deviceId + 1);
            }
            if (gpuAllocator_[deviceId] == nullptr) {
                std::string name = "gpu" + std::to_string(deviceId) + std::string("_pool");
                gpuAllocator_[deviceId] = new MemoryPool(new GpuAllocator());
            }
            return gpuAllocator_[deviceId];
        }
    }

    MemoryPool *MemoryManager::GetCpuMemoryPool() {
        {
            std::lock_guard<std::mutex> guard(mutex_);
            if (cpuMemoryPool_ != nullptr) {
                return cpuMemoryPool_;
            }
        }

        {
            std::lock_guard<std::mutex> guard(mutex_);
            if (cpuMemoryPool_ == nullptr) {
                cpuMemoryPool_ = new MemoryPool(new CpuAllocator());
            }
            return cpuMemoryPool_;

        }
    }

    MemoryManager::MemoryManager() {

    }
}
