//
// Created by Jarlene on 2018/2/11.
//

#ifndef MATRIX_INIT_H
#define MATRIX_INIT_H


#include "matrix/include/base/Tensor.h"
#include "matrix/include/store/MemoryManager.h"

namespace matrix {

    template<class T>
    inline void InitTensor(Tensor<T> tensor) {
        assert(tensor.Size() > 0);
        if (tensor.Data() == nullptr) {
            tensor.MutableData() = static_cast<T *>(MemoryManager::Global()->GetCpuMemoryPool()->dynamicAllocate(
                    tensor.Size() * sizeof(T)));
        }

    }

    template<class T>
    inline void DestoryTensor(Tensor<T> tensor) {
        if (tensor.Data() != nullptr) {
            MemoryManager::Global()->GetCpuMemoryPool()->freeMemory(tensor.MutableData(), tensor.Size() * sizeof(T));
        }
    }

}

#endif //MATRIX_INIT_H
