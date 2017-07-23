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
#elif defined(USE_BLAS)
#include <cblas.h>
#elif defined(USE_EIGEN)
#include <eigen3/Eigen/Eigen>
#endif



namespace matrix {

    enum BlasTranspose {
        NoTrans,
        Trans,
        ConjTrans
    };


    /// C := alpha*op(A)*op(B) + beta*C
    /// \tparam T  the type of input data
    /// \param TransA
    /// \param TransB
    /// \param M
    /// \param N
    /// \param K
    /// \param alpha
    /// \param A
    /// \param B
    /// \param beta
    /// \param C
    template <class T>
    inline void CPUGemm(const BlasTranspose TransA,
                        const BlasTranspose TransB, const int M, const int N, const int K,
                        const T alpha, const T *A, const T *B, const T beta,
                        T *C);


    template <>
    inline void CPUGemm<float>(const BlasTranspose TransA,
                               const BlasTranspose TransB, const int M, const int N, const int K,
                               const float alpha, const float *A, const float *B, const float beta,
                               float *C) {
        int lda = (TransA == NoTrans) ? K : M;
        int ldb = (TransB == NoTrans) ? N : K;
        CBLAS_TRANSPOSE Atrans, Btrans;
        switch (TransA) {
            case NoTrans:
                Atrans = CblasNoTrans;
                break;
            case Trans:
                Atrans = CblasTrans;
                break;
            case ConjTrans:
                Atrans = CblasConjTrans;
                break;
        }
        switch (TransB) {
            case NoTrans:
                Btrans = CblasNoTrans;
                break;
            case Trans:
                Btrans = CblasTrans;
                break;
            case ConjTrans:
                Btrans = CblasConjTrans;
                break;
        }
        cblas_sgemm(CblasRowMajor, Atrans, Btrans, M, N, K, alpha, A, lda, B,
                    ldb, beta, C, N);
    }

    template <>
    inline void CPUGemm<double>(const BlasTranspose TransA,
                                 const BlasTranspose TransB, const int M, const int N, const int K,
                                 const double alpha, const double *A, const double *B, const double beta,
                                 double *C) {
        int lda = (TransA == NoTrans) ? K : M;
        int ldb = (TransB == NoTrans) ? N : K;
        CBLAS_TRANSPOSE Atrans, Btrans;
        switch (TransA) {
            case NoTrans:
                Atrans = CblasNoTrans;
                break;
            case Trans:
                Atrans = CblasTrans;
                break;
            case ConjTrans:
                Atrans = CblasConjTrans;
                break;
        }
        switch (TransB) {
            case NoTrans:
                Btrans = CblasNoTrans;
                break;
            case Trans:
                Btrans = CblasTrans;
                break;
            case ConjTrans:
                Btrans = CblasConjTrans;
                break;
        }
        cblas_dgemm(CblasRowMajor, Atrans, Btrans, M, N, K, alpha, A, lda, B,
                    ldb, beta, C, N);
    }

    template <>
    inline void CPUGemm<int>(const BlasTranspose TransA,
                             const BlasTranspose TransB, const int M, const int N, const int K,
                             const int alpha, const int *A, const int *B, const int beta,
                             int *C) {

    }

    template <>
    inline void CPUGemm<long>(const BlasTranspose TransA,
                              const BlasTranspose TransB, const int M, const int N, const int K,
                              const long alpha, const long *A, const long *B, const long beta,
                              long *C) {

    }

    /// y := alpha*A*x + beta*y,   or   y := alpha*A^T*x + beta*y,
    /// \tparam T
    /// \param TransA
    /// \param M
    /// \param N
    /// \param alpha
    /// \param A
    /// \param x
    /// \param beta
    /// \param y
    template <class T>
    inline void CPUGemv(const BlasTranspose TransA, const int M, const int N,
                        const T alpha, const T *A, const T *x, const T beta,
                        T *y);

    template <>
    inline void CPUGemv<float>(const BlasTranspose TransA, const int M, const int N,
                               const float alpha, const float *A, const float *x, const float beta,
                               float *y) {
        CBLAS_TRANSPOSE Atrans;
        switch (TransA) {
            case NoTrans:
                Atrans = CblasNoTrans;
                break;
            case Trans:
                Atrans = CblasTrans;
                break;
            case ConjTrans:
                Atrans = CblasConjTrans;
                break;
            default:
                break;
        }
        cblas_sgemv(CblasRowMajor, Atrans, M, N, alpha, A, N, x, 1, beta, y, 1);
    }

    template <>
    inline void CPUGemv<double>(const BlasTranspose TransA, const int M, const int N,
                                const double alpha, const double *A, const double *x, const double beta,
                                double *y) {
        CBLAS_TRANSPOSE Atrans;
        switch (TransA) {
            case NoTrans:
                Atrans = CblasNoTrans;
                break;
            case Trans:
                Atrans = CblasTrans;
                break;
            case ConjTrans:
                Atrans = CblasConjTrans;
                break;
            default:
                break;
        }
        cblas_dgemv(CblasRowMajor, Atrans, M, N, alpha, A, N, x, 1, beta, y, 1);
    }


    template <>
    inline void CPUGemv<int>(const BlasTranspose TransA, const int M, const int N,
                             const int alpha, const int *A, const int *x, const int beta,
                             int *y) {

    }

    template <>
    inline void CPUGemv<long>(const BlasTranspose TransA, const int M, const int N,
                              const long alpha, const long *A, const long *x, const long beta,
                              long *y) {

    }

    /// Y = alpha * X + Y
    /// \tparam T
    /// \param N
    /// \param alpha
    /// \param X
    /// \param incx
    /// \param Y
    /// \param incy
    template <class T>
    inline void CPUAxpy(const int N, const T alpha, const T *X, int incx,
                        T *Y, int incy);

    template <>
    inline void CPUAxpy<float>(const int N, const float alpha, const float *X, int incx,
                               float *Y, int incy) {
        cblas_saxpy(N, alpha, X, incx, Y, incy);
    }

    template <>
    inline void CPUAxpy<double>(const int N, const double alpha, const double *X, int incx,
                                double *Y, int incy) {
        cblas_daxpy(N, alpha, X, incx, Y, incy);
    }

    template <>
    inline void CPUAxpy<int>(const int N, const int alpha, const int *X, int incx,
                             int *Y, int incy) {
    }

    template <>
    inline void CPUAxpy<long>(const int N, const long alpha, const long *X, int incx,
                              long *Y, int incy) {
    }

}

#endif //MATRIX_MATH_H
