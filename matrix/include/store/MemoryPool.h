//
// Created by Jarlene on 2017/8/9.
//

#ifndef MATRIX_MEMORYPOOL_H
#define MATRIX_MEMORYPOOL_H

#include <string>
#include <map>
#include <vector>
#include <mutex>
#include "Allocator.h"

namespace matrix {
    class MemoryPool {
    public:
        /// if sizeLimit == 0, the pool allocator is a simple wrapper of allocator.
        /// \param allocator
        /// \param sizeLimit
        /// \param name
        MemoryPool(Allocator *allocator,
                   size_t sizeLimit = 0,
                   const std::string &name = "pool");


        ///
        /// \param size allocate memeory size
        /// \return return the address of the memory
        void *dynamicAllocate(size_t size);

        /// static allocate graph node according to the color.
        /// \param colorSize
        void staticAllocate(const std::map<int, size_t> &colorSize);

        /// free static allocate memory
        /// \param color
        void freeMemory(int color);

        /// free dynamic allocate memory
        /// \param color
        void freeMemory(void* ptr, size_t n);

        /// destructor
        ~MemoryPool();

        /// get static memory from color
        /// \param color
        /// \return
        void * getMemory(int color);


        void PrintMemory();


        void freeAll();

    private:
        std::mutex mutex_;

        std::map<size_t, std::vector<void *>> pool_;
        std::unique_ptr<Allocator> allocator_;

        std::map<int, void *> memroy_;

        std::vector<void*> pools;

        size_t sizeLimit_;
        size_t poolMemorySize_;
        std::string name_;
    };
}

#endif //MATRIX_MEMORYPOOL_H
