//
// Created by Jarlene on 2017/8/9.
//

#ifndef MATRIX_ALLOCATOR_H
#define MATRIX_ALLOCATOR_H

#include <string>
#include "matrix/include/utils/Logger.h"

namespace matrix {
    class Allocator {
    public:
        virtual ~Allocator() {}
        virtual void* alloc(size_t size) = 0;
        virtual void free(void* ptr) = 0;
        virtual std::string getName() = 0;

    public:
        const static size_t alignment_ = 16;
    };
}

#endif //MATRIX_ALLOCATOR_H
