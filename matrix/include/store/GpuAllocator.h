//
// Created by Jarlene on 2017/8/9.
//

#ifndef MATRIX_GPUALLOCATOR_H
#define MATRIX_GPUALLOCATOR_H

#include "Allocator.h"

namespace matrix {
    class GpuAllocator : public Allocator {
    public:
        ~GpuAllocator();

        virtual void* alloc(size_t size);

        virtual void free(void* ptr);

        virtual std::string getName();

    };
}

#endif //MATRIX_GPUALLOCATOR_H
