//
// Created by Jarlene on 2017/3/20.
//

#include "matrix/include/store/GpuAllocator.h"

namespace matrix {

    GpuAllocator::~GpuAllocator() {

    }

    void *GpuAllocator::alloc(size_t size) {
        return nullptr;
    }

    void GpuAllocator::free(void *ptr) {

    }

    std::string GpuAllocator::getName() {
        return "GpuAllocator";
    }
}