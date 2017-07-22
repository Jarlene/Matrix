//
// Created by Jarlene on 2017/7/21.
//

#ifndef MATRIX_MATH_H
#define MATRIX_MATH_H

#include <math.h>
#include <assert.h>
#include <functional>
#include <random>
#include <sys/time.h>

#ifdef USE_MP
#include <omp.h>
#endif
#ifdef USE_MKL
#include <mkl.h>
#include <mkl_cblas.h>
#include <mkl_vsl.h>
#include <mkl_vsl_functions.h>
#elif USE_BLAS
#include <cblas.h>
#endif


namespace matrix {



}

#endif //MATRIX_MATH_H
