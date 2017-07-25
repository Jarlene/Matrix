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


    /// Y=X
    /// \tparam T
    /// \param N
    /// \param x
    /// \param incx
    /// \param y
    /// \param incy
    template <class T>
    inline void CPUCopy(const int N, T* x, int incx, T* y, int incy);


    template <>
    inline void CPUCopy<float>(const int N, float* x, int incx, float* y, int incy) {
        cblas_scopy(N, x, incx, y, incy);
    }

    template <>
    inline void CPUCopy<double>(const int N, double* x, int incx, double* y, int incy) {
        cblas_dcopy(N, x, incx, y, incy);
    }


    template <>
    inline void CPUCopy<int>(const int N, int* x, int incx, int* y, int incy) {
        int posx = 0;
        int posy = 0;
#ifdef USE_MP
        omp_set_num_threads(CPU_CORES);
#pragma omp parallel for
#endif
        for (int i = 0; i < N; ++i) {
            y[posy] = x[posx];
            posy += incy;
            posx += incx;
        }
    }

    template <>
    inline void CPUCopy<long>(const int N, long* x, int incx, long* y, int incy) {
        int posx = 0;
        int posy = 0;
#ifdef USE_MP
        omp_set_num_threads(CPU_CORES);
#pragma omp parallel for
#endif
        for (int i = 0; i < N; ++i) {
            y[posy] = x[posx];
            posy += incy;
            posx += incx;
        }
    }


    ///
    /// \tparam T
    /// \param N
    /// \param x
    /// \param incx
    /// \param y
    /// \param incy
    template <class T>
    inline void CPUSwap(const int N, T * x, int incx, T *y, int incy );

    template <>
    inline void CPUSwap<float>(const int N, float * x, int incx, float *y, int incy ) {
        cblas_sswap(N, x, incx, y, incy);
    }

    template <>
    inline void CPUSwap<double>(const int N, double * x, int incx, double *y, int incy ) {
        cblas_dswap(N, x, incx, y, incy);
    }

    template <class T>
    inline void CPUSwap(const int N, T * x) {
#ifdef USE_MP
        omp_set_num_threads(CPU_CORES);
#pragma omp parallel for
#endif
        for (int i = 0; i < N/2; ++i) {
            std::swap(x[i], x[N-1-i]);
        }
    }




    /// res = x'*y
    /// \tparam T
    /// \param N
    /// \param x
    /// \param y
    /// \param res
    template <class T>
    inline void CPUDot(const int N, const T* x, const T* y, T& res);


    template <>
    inline void CPUDot<float>(const int N, const float* x, const float* y, float& res) {
        res = cblas_sdot(N, x, 1, y, 1);
    }

    template <>
    inline void CPUDot<double>(const int N, const double* x, const double* y, double& res) {
        res = cblas_ddot(N, x, 1, y, 1);
    }

    template <>
    inline void CPUDot<int>(const int N, const int* x, const int* y, int& res) {
    }

    template <>
    inline void CPUDot<long>(const int N, const long* x, const long* y, long& res) {
    }



    template <class T>
    inline void CPUAdd(const int N, const T *a, const T *b, T *y) {
#ifdef USE_MP
#pragma omp parallel for
#endif
        for (int i=0; i < N; ++i) {
            y[i] = a[i] + b[i];
        }
    }


    template <class T>
    inline void CPUSub(const int N, const T *a, const T *b, T *y) {
#ifdef USE_MP
#pragma omp parallel for
#endif
        for (int i=0; i < N; ++i) {
            y[i] = a[i] - b[i];
        }
    }


    template <class T>
    inline void CPUMul(const int N, const T *a, const T *b, T *y) {
#ifdef USE_MP
#pragma omp parallel for
#endif
        for (int i=0; i < N; ++i) {
            y[i] = a[i] * b[i];
        }
    }


    template <class T>
    inline void CPUDiv(const int N, const T *a, const T *b, T *y) {
#ifdef USE_MP
#pragma omp parallel for
#endif
        for (int i=0; i < N; ++i) {
            y[i] = a[i] / b[i];
        }
    }


    template <class T>
    inline void CPUTanh(const int N, const T *x, T *y) {
#ifdef USE_MP
#pragma omp parallel for
#endif
        for (int i=0; i < N; ++i) {
            y[i] = (exp(x[i])-exp(-x[i]))/(exp(x[i]) + exp(-x[i]));
        }
    }


    template <class T>
    inline void CPUTanhGrad(const int N,  T*x, T *y,  T *z) {
#ifdef USE_MP
#pragma omp parallel for
#endif
        for (int i=0; i < N; ++i) {
            z[i] = y[i] * (T(1) - x[i]*x[i]);
        }
    }


    template <class T>
    inline void Sigmoid(const int N,  T*x, T *y) {
#ifdef USE_MP
#pragma omp parallel for
#endif
        for (int i=0; i < N; ++i) {
            y[i] = T(1)/(T(1) + exp(T(-1) * x[i]));
        }
    }


    template <class T>
    inline void SigmoidGrad(const int N, T *x, T *y,  T *z) {
#ifdef USE_MP
#pragma omp parallel for
#endif
        for (int i=0; i < N; ++i) {
            z[i] = y[i] * x[i]*((T)1-x[i]);
        }
    }


    template <class T>
    inline void Relu(const int N, T *x, T *y) {
#ifdef USE_MP
#pragma omp parallel for
#endif
        for (int i=0; i < N; ++i) {
            y[i] = (x[i]>(T)0 ? x[i] : 0);
        }
    }

    template <class T>
    inline void ReluGrad(const int N, T *dx, T *x, T* dy) {
#ifdef USE_MP
#pragma omp parallel for
#endif
        for (int i=0; i < N; ++i) {
            dx[i] = (x[i] > (T)0 ? dy[i] : 0);
        }
    }



    template <class T>
    inline void Softmax(const int N, T* x, T* y) {
        T max = x[0];
#ifdef USE_MP
#pragma omp parallel for
#endif
        for (int i=1; i<N; ++i) {
            if (max < x[i]) {
                max = x[i];
            }
        }

        T sum = (T)0;

#ifdef USE_MP
#pragma omp parallel for
#endif
        for (int i=0; i<N; ++i) {
            y[i] = std::exp(x[i] - max);
            sum += y[i];
        }
#ifdef USE_MP
#pragma omp parallel for
#endif
        for (int i=0; i<N; ++i) {
            y[i] /= sum;
        }
    }


    /// cross-entropy
    /// \tparam T
    /// \param N prediction data length
    /// \param in1 prediction value
    /// \param M real data length
    /// \param in2 real value
    /// \param out
    template <class T>
    inline void CrossEntropy(const int N, T *in1, const int M, T *in2, T *out) {
        if (N == M) {
#ifdef USE_MP
#pragma omp parallel for
#endif
            for (int i = 0; i < N; ++i) {
                out[0] += T(-1) * in2[i] * log(in1[i]);
            }

        } else {
#ifdef USE_MP
#pragma omp parallel for
#endif
            for (int i = 0; i < M; ++i) {
                for (int j = 0; j < N / M; ++j) {
                    out[0] += T(-1) * in2[i] * log(in1[i * M + j]);
                }
            }
        }
        out[0] /= N;
    }


    /// cross-entropy gradient
    /// \tparam T
    /// \param N prediction data length
    /// \param in1 prediction value
    /// \param M real data length
    /// \param in2 real value
    /// \param out
    template <class T>
    inline void CrossEntropyGrad(const int N, T *in1, const int M, T *in2, T *out) {
        if (N == M) {
#ifdef USE_MP
#pragma omp parallel for
#endif
            for (int i = 0; i < N; ++i) {
                out[i] = -in2[i] / in1[i];
            }
        } else {
#ifdef USE_MP
#pragma omp parallel for
#endif
            for (int i = 0; i < M; ++i) {
                for (int j = 0; j < N / M; ++j) {
                    out[i * M + j] = -in2[i] / std::log(in1[i * M + j]);
                }
            }
        }
    }
}

#endif //MATRIX_MATH_H
