//
// Created by Jarlene on 2017/8/9.
//

#ifndef MATRIX_MEMORYMANAGER_H
#define MATRIX_MEMORYMANAGER_H

#include "MemoryPool.h"

namespace matrix {
    class MemoryManager {
    public:
        static MemoryManager *Global() {
            static MemoryManager manager;
            return &manager;
        }

        MemoryPool * GetCpuMemoryPool();

        MemoryPool * GetGpuMemoryPool(int deviceId);

    private:
        MemoryManager();

    private:
        MemoryPool* cpuMemoryPool_;
        std::mutex mutex_;
        std::vector<MemoryPool*> gpuAllocator_;
    };
}

#endif //MATRIX_MEMORYMANAGER_H
