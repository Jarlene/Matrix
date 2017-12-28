//
// Created by Jarlene on 2017/12/28.
//

#ifndef MATRIX_CUDA_H
#define MATRIX_CUDA_H

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cublas.h>
#include <curand.h>
#include <cuda.h>


namespace matrix {
static const int CUDA_NUM_THREADS = 1024;
#define MAX_GPUS 8;

#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < n; i += blockDim.x * gridDim.x)

}

#endif


#endif //MATRIX_CUDA_H
