//
// Created by Jarlene on 2017/3/20.
//

#include "matrix/include/store/CpuAllocator.h"
#include <mm_malloc.h>

namespace matrix {

    CpuAllocator::~CpuAllocator() {

    }

    void *CpuAllocator::alloc(size_t size) {
        void* ptr = _mm_malloc(size, alignment_);
        if (ptr == nullptr) {
            std::string msg = getName() +  " allocate memory error \n";
            Logger::Global()->Fatal(msg.c_str());
        }
        return ptr;
    }

    void CpuAllocator::free(void *ptr) {
        if (ptr) {
            ::free(ptr);
            ptr = nullptr;
        }
    }

    std::string CpuAllocator::getName() {
        return "CpuAllocator";
    }
}