//
// Created by Jarlene on 2017/8/9.
//

#ifndef MATRIX_CPUALLOCATOR_H
#define MATRIX_CPUALLOCATOR_H

#include "Allocator.h"

namespace matrix {
    class CpuAllocator : public Allocator {
    public:
        ~CpuAllocator();

        virtual void* alloc(size_t size);

        virtual void free(void* ptr);

        virtual std::string getName();
    };
}

#endif //MATRIX_CPUALLOCATOR_H
