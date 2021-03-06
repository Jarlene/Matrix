//
// Created by Jarlene on 2017/7/21.
//

#ifndef MATRIX_MATH_H
#define MATRIX_MATH_H

#include <math.h>
#include <assert.h>
#include <functional>
#include <vector>
#include <random>
#include <sys/time.h>
#include "Logger.h"
#include "Eigen.h"
#ifdef USE_MP
#include <omp.h>
#endif


#ifdef USE_MKL
#ifndef BLAS
#define BLAS
#endif
#include <mkl.h>
#include <mkl_cblas.h>
#include <mkl_vsl.h>
#include <mkl_vsl_functions.h>
#elif defined(USE_BLAS)
#ifndef BLAS
#define BLAS
#endif
#include <cblas.h>
#endif



namespace matrix {


    static inline bool isLess(int a, int b) {
        return static_cast<unsigned>(a) < static_cast<unsigned>(b);
    }


    static struct timeval tv;
    static std::mt19937 rnd_engine_;

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
#ifdef BLAS
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
#elif defined(USE_EIGEN)
        int lda = (TransA == NoTrans) ? M : K; // A 的行
        int ldb = (TransB == NoTrans) ? N : K; // B 的列
        int aCol = (TransA == NoTrans) ? K : M; // A的列
        auto aMatrix = create<float>(A, lda,  aCol);
        auto bMatrix = create<float>(B, aCol, ldb);
        auto cMatrix = create<float>(C, lda, ldb);
        cMatrix = alpha * aMatrix * bMatrix + beta * cMatrix;
#endif
    }

    template <>
    inline void CPUGemm<double>(const BlasTranspose TransA,
                                 const BlasTranspose TransB, const int M, const int N, const int K,
                                 const double alpha, const double *A, const double *B, const double beta,
                                 double *C) {
#ifdef BLAS
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
#elif defined(USE_EIGEN)
        int lda = (TransA == NoTrans) ? M : K; // A 的行
        int ldb = (TransB == NoTrans) ? K : N; // B 的列
        int aCol = (TransA == NoTrans) ? K : M; // A的列
        auto aMatrix = create<double>(A, lda,  aCol);
        auto bMatrix = create<double>(B, aCol, ldb);
        auto cMatrix = create<double>(C, lda, ldb);
        cMatrix = alpha * aMatrix * bMatrix + beta * cMatrix;
#endif
    }

    template <>
    inline void CPUGemm<int>(const BlasTranspose TransA,
                             const BlasTranspose TransB, const int M, const int N, const int K,
                             const int alpha, const int *A, const int *B, const int beta,
                             int *C) {
#ifdef USE_EIGEN
        int lda = (TransA == NoTrans) ? M : K; // A 的行
        int ldb = (TransB == NoTrans) ? N : K; // B 的列
        int aCol = (TransA == NoTrans) ? K : M; // A的列
        auto aMatrix = create<int>(A, lda,  aCol);
        auto bMatrix = create<int>(B, aCol, ldb);
        auto cMatrix = create<int>(C, lda, ldb);
        cMatrix = alpha * aMatrix * bMatrix + beta * cMatrix;
#endif
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
#ifdef BLAS
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
#elif defined(USE_EIGEN)
        int lda = (TransA == NoTrans)? M : N;
        int cda = (TransA == NoTrans)? N : M;
        auto aMatrix = create<>(A, lda, cda);
        auto xVector = create<>(x, cda);
        auto yVector = create<>(y, lda);
        yVector = alpha * aMatrix * xVector + beta * yVector;
#endif
    }

    template <>
    inline void CPUGemv<double>(const BlasTranspose TransA, const int M, const int N,
                                const double alpha, const double *A, const double *x, const double beta,
                                double *y) {
#ifdef BLAS
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
#elif defined(USE_EIGEN)
        int lda = (TransA == NoTrans)? M : N;
        int cda = (TransA == NoTrans)? N : M;
        auto aMatrix = create<double>(A, lda, cda);
        auto xVector = create<double>(x, cda);
        auto yVector = create<double>(y, lda);
        yVector = alpha * aMatrix * xVector + beta * yVector;
#endif
    }


    template <>
    inline void CPUGemv<int>(const BlasTranspose TransA, const int M, const int N,
                             const int alpha, const int *A, const int *x, const int beta,
                             int *y) {
#ifdef USE_EIGEN
        int lda = (TransA == NoTrans)? M : N;
        int cda = (TransA == NoTrans)? N : M;
        auto aMatrix = create<int>(A, lda, cda);
        auto xVector = create<int>(x, cda);
        auto yVector = create<int>(y, lda);
        yVector = alpha * aMatrix * xVector + beta * yVector;
#endif
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
#ifdef BLAS
        cblas_saxpy(N, alpha, X, incx, Y, incy);
#else
        int posx = 0;
        int posy = 0;
#ifdef USE_MP
        omp_set_num_threads(CPU_CORES);
#pragma omp parallel for
#endif
        for (int i = 0; i <N; ++i) {
            Y[posy] += alpha * X[posx];
            posx += incx;
            posy += incy;
        }
#endif
    }

    template <>
    inline void CPUAxpy<double>(const int N, const double alpha, const double *X, int incx,
                                double *Y, int incy) {
#ifdef BLAS
        cblas_daxpy(N, alpha, X, incx, Y, incy);
#else
        int posx = 0;
        int posy = 0;
#ifdef USE_MP
        omp_set_num_threads(CPU_CORES);
#pragma omp parallel for
#endif
        for (int i = 0; i <N; ++i) {
            Y[posy] += alpha * X[posx];
            posx += incx;
            posy += incy;
        }
#endif
    }

    template <>
    inline void CPUAxpy<int>(const int N, const int alpha, const int *X, int incx,
                             int *Y, int incy) {
        int posx = 0;
        int posy = 0;
#ifdef USE_MP
        omp_set_num_threads(CPU_CORES);
#pragma omp parallel for
#endif
        for (int i = 0; i <N; ++i) {
            Y[posy] += alpha * X[posx];
            posx += incx;
            posy += incy;

        }
    }

    template <>
    inline void CPUAxpy<long>(const int N, const long alpha, const long *X, int incx,
                              long *Y, int incy) {
        int posx = 0;
        int posy = 0;
#ifdef USE_MP
        omp_set_num_threads(CPU_CORES);
#pragma omp parallel for
#endif
        for (int i = 0; i <N; ++i) {
            Y[posy] += alpha * X[posx];
            posx += incx;
            posy += incy;
        }
    }
    /**
     * Y = alpha * X + beta * Y
     * @param T
     * @param N
     * @param alpha
     * @param X
     * @param beta
     * @param Y
     */
    template <class T>
    inline void CPUAxpby(const int N, const T alpha, const T *X, int incx,
                         const T beta, T *Y, int incy);

    template <>
    inline void CPUAxpby<float>(const int N, const float alpha, const float *X, int incx,
                         const float beta, float *Y, int incy) {
#ifdef BLAS
        cblas_saxpby(N, alpha, X, incx, beta, Y, incy);
#endif
    }

    template <>
    inline void CPUAxpby<double>(const int N, const double alpha, const double *X, int incx,
                                const double beta, double *Y, int incy) {
#ifdef BLAS
        cblas_daxpby(N, alpha, X, incx, beta, Y, incy);
#endif
    }


    template <>
    inline void CPUAxpby<int>(const int N, const int alpha, const int *X, int incx,
                                 const int beta, int *Y, int incy) {

    }
    template <>
    inline void CPUAxpby<long>(const int N, const long alpha, const long *X, int incx,
                              const long beta, long *Y, int incy) {

    }

    /// Y=X
    /// \tparam T
    /// \param N
    /// \param x
    /// \param incx
    /// \param y
    /// \param incy
    template <class T>
    inline void CPUCopy(const int N, const T* x, int incx, T* y, int incy);


    template <>
    inline void CPUCopy<float>(const int N, const float* x, int incx, float* y, int incy) {
#ifdef BLAS
        cblas_scopy(N, x, incx, y, incy);
#else
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
#endif
    }

    template <>
    inline void CPUCopy<double>(const int N, const double* x, int incx, double* y, int incy) {
#ifdef BLAS
        cblas_dcopy(N, x, incx, y, incy);
#else
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
#endif
    }


    template <>
    inline void CPUCopy<int>(const int N, const int* x, int incx, int* y, int incy) {
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
    inline void CPUCopy<long>(const int N, const long* x, int incx, long* y, int incy) {
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
#ifdef BLAS
        cblas_sswap(N, x, incx, y, incy);
#else
        int posx = 0, posy = 0;
#ifdef USE_MP
        omp_set_num_threads(CPU_CORES);
#pragma omp parallel for
#endif
        for (int i = 0; i < N; ++i) {
            std::swap(x[posx], y[posy]);
            posx += incx;
            posy += incy;
        }
#endif
    }

    template <>
    inline void CPUSwap<double>(const int N, double * x, int incx, double *y, int incy ) {
#ifdef BLAS
        cblas_dswap(N, x, incx, y, incy);
#else
        int posx = 0, posy = 0;
#ifdef USE_MP
        omp_set_num_threads(CPU_CORES);
#pragma omp parallel for
#endif
        for (int i = 0; i < N; ++i) {
            std::swap(x[posx], y[posy]);
            posx += incx;
            posy += incy;
        }
#endif
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
#ifdef BLAS
        res = cblas_sdot(N, x, 1, y, 1);
#else
#ifdef USE_MP
        omp_set_num_threads(CPU_CORES);
#pragma omp parallel for
#endif
        for (int i = 0; i < N; ++i) {
            res += x[i] * y[i];
        }
#endif
    }

    template <>
    inline void CPUDot<double>(const int N, const double* x, const double* y, double& res) {
#ifdef BLAS
        res = cblas_ddot(N, x, 1, y, 1);
#else
#ifdef USE_MP
        omp_set_num_threads(CPU_CORES);
#pragma omp parallel for
#endif
        for (int i = 0; i < N; ++i) {
            res += x[i] * y[i];
        }
#endif
    }

    template <>
    inline void CPUDot<int>(const int N, const int* x, const int* y, int& res) {
#ifdef USE_MP
        omp_set_num_threads(CPU_CORES);
#pragma omp parallel for
#endif
        for (int i = 0; i < N; ++i) {
            res += x[i] * y[i];
        }
    }

    template <>
    inline void CPUDot<long>(const int N, const long* x, const long* y, long& res) {
#ifdef USE_MP
        omp_set_num_threads(CPU_CORES);
#pragma omp parallel for
#endif
        for (int i = 0; i < N; ++i) {
            res += x[i] * y[i];
        }
    }


    template <class T>
    inline void Value(const int N, T* out, T val) {
        if (val == T(0)) {
            memset(out, 0, sizeof(T) * N);
            return;
        }

#ifdef USE_EIGEN
        Vec<T> vec = create<T>(out, N);
        vec.fill(val);
#else
#ifdef USE_MP
#pragma omp parallel for
#endif
        for (int i = 0; i < N; ++i) {
            out[i] = val;
        }
#endif
    }

    template <class T>
    inline void Scale(const int N, T* out, T val);

    template <>
    inline void Scale<float>(const int N, float* out, float val) {
#ifdef BLAS
        cblas_sscal(N, val, out, 1);
#elif define(USE_EIGEN)
        auto v = create<float>(out, N);
        v = v * val;
#else
#ifdef USE_MP
#pragma omp parallel for
#endif
        for (int i = 0; i < N; ++i) {
            out[i] *= val;
        }
#endif
    }

    template <>
    inline void Scale<double>(const int N, double* out, double val) {
#ifdef BLAS
        cblas_dscal(N, val, out, 1);

#elif define(USE_EIGEN)
        auto v = create<double>(out, N);
        v = v*val;
#else
        #ifdef USE_MP
#pragma omp parallel for
#endif
        for (int i = 0; i < N; ++i) {
            out[i] *= val;
        }
#endif
    }

    template <>
    inline void Scale<int>(const int N, int* out, int val) {
#ifdef USE_EIGEN
        auto v = create<int>(out, N);
        v *= val;
#else
#ifdef USE_MP
#pragma omp parallel for
#endif
        for (int i = 0; i < N; ++i) {
            out[i] *= val;
        }
#endif
    }

    template <>
    inline void Scale<long>(const int N, long* out, long val) {
#ifdef USE_MP
#pragma omp parallel for
#endif
        for (int i = 0; i < N; ++i) {
            out[i] *= val;
        }
    }

    template <class T>
    inline void Random(const int N, T *out, T mu, T sigma) {
        gettimeofday(&tv,NULL);
        std::normal_distribution<T> dist_normal(mu, sigma);
        rnd_engine_.seed((unsigned int) (tv.tv_sec * 1000 * 1000 + tv.tv_usec));
#ifdef USE_MP
#pragma omp parallel for
#endif
        for (int i = 0; i < N; ++i) {
            out[i] = dist_normal(rnd_engine_);
        }
    }


    template <class T>
    inline void Add(const int N, const int M, const T *a,  const T *b, T *y) {
#ifdef USE_EIGEN
        auto av = create<T>(a, N, M);
        auto bv = create<T>(b, N);
        auto yv = create<T>(y, N, M);
        yv = av.colwise() + bv;

#else
#ifdef USE_MP
#pragma omp parallel for
#endif
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < M; ++j) {
                y[i * M +j] = a[i * M +j] + b[i];
            }
        }
#endif
    }

    template <class T>
    inline void Add(const int N, const T *a, const T *b, T *y) {
#ifdef USE_EIGEN
        auto av = create<T>(a, N);
        auto bv = create<T>(b, N);
        auto yv = create<T>(y, N);
        yv = (av + bv);
#else
#ifdef USE_MP
#pragma omp parallel for
#endif
        for (int i=0; i < N; ++i) {
            y[i] = a[i] + b[i];
        }
#endif
    }


    template <class T>
    inline void Sub(const int N, const T *a, const T *b, T *y) {
#ifdef USE_EIGEN
        auto av = create<T>(a, N);
        auto bv = create<T>(b, N);
        auto yv = create<T>(y, N);
        yv = av - bv;
#else
#ifdef USE_MP
#pragma omp parallel for
#endif
        for (int i=0; i < N; ++i) {
            y[i] = a[i] - b[i];
        }
#endif
    }


    template <class T>
    inline void Mul(const int N, const T *a, const T *b, T *y) {
#ifdef USE_EIGEN
        auto av = create<T>(a, N);
        auto bv = create<T>(b, N);
        auto yv = create<T>(y, N);
        yv = av.array() * bv.array();
#else
#ifdef USE_MP
#pragma omp parallel for
#endif
        for (int i=0; i < N; ++i) {
            y[i] = a[i] * b[i];
        }
#endif
    }


    template <class T>
    inline void Div(const int N, const T *a, const T *b, T *y) {
#ifdef USE_EIGEN
        auto av = create<T>(a, N);
        auto bv = create<T>(b, N);
        auto yv = create<T>(y, N);
        yv = av.array() / bv.array();
#else
#ifdef USE_MP
#pragma omp parallel for
#endif
        for (int i=0; i < N; ++i) {
            y[i] = a[i] / b[i];
        }
#endif
    }

    template <class T>
    inline void Reciprocal(const int N,  T *x) {
#ifdef USE_EIGEN
        auto xv = create<T>(x, N);
        xv /= T(1.0);
#else
#ifdef USE_MP
#pragma omp parallel for
#endif
        for (int i=0; i < N; ++i) {
            x[i] = T(1.0) / x[i];
        }
#endif
    }

    template <class T>
    inline void Negative(const int N,  T *x) {
#ifdef USE_EIGEN
        auto xv = create<T>(x, N);
        xv = (T(0) - xv);
#else
#ifdef USE_MP
#pragma omp parallel for
#endif
        for (int i=0; i < N; ++i) {
            x[i] = -x[i];
        }
#endif
    }




    /// tanh
    /// \tparam T
    /// \param N
    /// \param x
    /// \param y
    template <class T>
    inline void Tanh(const int N, const T *x, T *y) {
#ifdef USE_MP
#pragma omp parallel for
#endif
        for (int i=0; i < N; ++i) {
            y[i] = (exp(x[i])-exp(-x[i]))/(exp(x[i]) + exp(-x[i]));
        }
    }


    /// tanh gradient
    /// \tparam T
    /// \param N
    /// \param x
    /// \param y
    /// \param z
    template <class T>
    inline void TanhGrad(const int N,  const T *x,  const T *y, T *z) {
#ifdef USE_MP
#pragma omp parallel for
#endif
        for (int i=0; i < N; ++i) {
            z[i] =  y[i] * (T(1) - x[i]*x[i]);
        }
    }


    /// sigmoid
    /// \tparam T
    /// \param N
    /// \param x
    /// \param y
    template <class T>
    inline void Sigmoid(const int N,  const T*x, T *y) {
#ifdef USE_MP
#pragma omp parallel for
#endif
        for (int i=0; i < N; ++i) {
            y[i] = T(1)/(T(1) + exp(T(-1) * x[i]));
        }
    }


    /// sigmoid gradient
    /// \tparam T
    /// \param N
    /// \param x
    /// \param y
    /// \param z
    template <class T>
    inline void SigmoidGrad(const int N, const T *x,  const T *y, T *z) {
#ifdef USE_MP
#pragma omp parallel for
#endif
        for (int i=0; i < N; ++i) {
            z[i] =  y[i]*x[i]*((T)1-x[i]);
        }
    }



    /// relu
    /// \tparam T
    /// \param N
    /// \param x
    /// \param y
    template <class T>
    inline void Relu(const int N, const T *x, T *y) {
#ifdef USE_MP
#pragma omp parallel for
#endif
        for (int i=0; i < N; ++i) {
            y[i] = (x[i] > T(0) ? x[i] : T(0));
        }
    }

    /// relu gradient
    /// \tparam T
    /// \param N
    /// \param dx
    /// \param x
    /// \param dy
    template <class T>
    inline void ReluGrad(const int N, const T *x, const T *dx,  T* dy) {
#ifdef USE_MP
#pragma omp parallel for
#endif
        for (int i=0; i < N; ++i) {
            dy[i] = (x[i] > (T)0 ? dx[i] : 0);
        }
    }



    /// softmax
    /// \tparam T
    /// \param N
    /// \param x
    /// \param y
    template <class T>
    inline void Softmax(const int N, const T* x, T* y) {
        const T max = *std::max_element(x, x + N);
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




    template <class T>
    inline void SoftmaxGrad(const int N, const int D, const T* x, const T* pre,  T* y) {

#ifdef USE_MP
#pragma omp parallel for
#endif

        for (int i = 0; i < N; ++i) {

            T sum = T(0);
            for (int j = 0; j < D; ++j) {
                sum += x[i * D + j] * pre[i * D + j];
            }

            for (int k = 0; k < D; ++k) {
                y[i * D + k] = x[i * D + k] * (pre[i * D + k] - sum);
            }
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
    inline void CrossEntropy(const int N, const T *in1, const int M, const T *in2, T *out) {
        int class_num = N / M;
#ifdef USE_MP
#pragma omp parallel for
#endif
            for (int i = 0; i < M; ++i) {
                int label = (int)in2[i];
                int index = i * class_num + label;
                out[0] += T(-1) * log(in1[index]);
            }
        out[0] /= M;
    }


    /// cross-entropy gradient
    /// \tparam T
    /// \param N prediction data length
    /// \param in1 prediction value
    /// \param M real data length
    /// \param in2 real value
    /// \param out
    template <class T>
    inline void CrossEntropyGrad(const int N, const T *in1, const int M, const T *in2, T *out) {

        int class_num = N / M;
#ifdef USE_MP
#pragma omp parallel for
#endif
        for (int i = 0; i < M; ++i) {
            int label = (int)in2[i];
            int index = i * class_num + label;
            out[index] = T(-1.0) / in1[index];
        }
    }


    /// rms loss
    /// \tparam T
    /// \param N  prediction data length
    /// \param in1 prediction value
    /// \param M label data length
    /// \param in2 label value
    /// \param out
    template <class T>
    inline void RMSLoss(const int N, const T *in1, const int M, const T *in2, T *out) {
        if (N == M) {
#ifdef USE_MP
#pragma omp parallel for
#endif
            for (int i = 0; i < N; ++i) {
                out[0] += T(0.5) * (in1[i] - in2[i]) * (in1[i] - in2[i]);
            }
        } else {
#ifdef USE_MP
#pragma omp parallel for
#endif
            for (int i = 0; i < M; ++i) {
                for (int j = 0; j < N / M; ++j) {
                    int idx = static_cast<int>(in2[i]);
                    if (j == idx) {
                        out[0] += T(0.5) * (in1[i] - 1) * (in1[i] - 1);
                    } else {
                        out[0] += T(0.5) * in1[i] * in1[i];
                    }
                }
            }
        }
        out[0] /= M;
    }


    /// rms loss grad
    /// \tparam T
    /// \param N prediction data length
    /// \param in1 prediction value
    /// \param M  label data length
    /// \param in2 label value
    /// \param out
    template <class T>
    inline void RMSLossGrad(const int N, const T *in1, const int M, const T *in2, T *out) {
        if (N == M) {
#ifdef USE_MP
#pragma omp parallel for
#endif
            for (int i = 0; i < N; ++i) {
                out[i] = (in1[i] - in2[i]);
            }
        } else {
#ifdef USE_MP
#pragma omp parallel for
#endif
            for (int i = 0; i < M; ++i) {
                for (int j = 0; j < N / M; ++j) {
                    int idx = static_cast<int>(in2[i]);
                    if (j == idx) {
                        out[i * M + j] = (in1[i * M + j] - 1);
                    } else {
                        out[i * M + j] = in1[i * M + j];
                    }
                }
            }
        }
    }


    template <class T>
    inline void SoftmaxCrossEntropy(const int N, const T *data, const int M, const T *label, T *out) {
        int class_num = N/M;
#ifdef USE_MP
#pragma omp parallel for
#endif
        for (int i = 0; i < M; ++i) {

            T sum = T(0.0);
            for (int j = 0; j < class_num; ++j) {
                sum += exp(data[i * class_num +j]);
            }

            out[0] += log(sum) - data[static_cast<int>(label[i])];
        }

        out[0] /= M;
    }

    template <class T>
    inline void SoftmaxCrossEntropyGrad(const int N, const T *data, const int M, const T *label, T *out) {
        int class_num = N / M;
#ifdef USE_MP
#pragma omp parallel for
#endif
        for (int i = 0; i < M; ++i) {
            const T *d = data + i * class_num;
            T *o = out + i * class_num;
            Softmax<T>(class_num, d, o);
            o[static_cast<int>(label[i])] -= 1;
        }
    }


    template <class T>
    inline void Reduce(const int N, std::function<void(int)> func) {
#ifdef USE_MP
#pragma omp parallel for
#endif
        for (int i = 0; i < N; ++i) {
            func(i);
        }
    }


    template <class T>
    inline void SumCopy(const int N, const T *in, const int M, T *out) {
#ifdef USE_MP
#pragma omp parallel for
#endif
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N / M; ++j) {
                out[i] += in[i * N / M + j];
            }
        }
    }


    template <class T, int order>
    inline void Img2Col(const T *input, const int channels, const int height, const int width,
                        const int kernel_h, const int kernel_w, const int dilation_h,
                        const int dilation_w, const int pad_t, const int pad_l,
                        const int pad_b, const int pad_r, const int stride_h, const int stride_w,
                        T *output) {
        if (order == 0) {
            const int output_h = (height + pad_b + pad_t - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
            const int output_w = (width + pad_l + pad_r - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

            // padding = 0; dilation = 1;
            if (dilation_h == 1 && dilation_w == 1 && pad_l == 0 && pad_r == 0 &&
                pad_t == 0 && pad_b == 0) {
                for (auto k = 0; k < channels * kernel_h * kernel_w; k++) {
                    const auto nip = k / (kernel_h * kernel_w);
                    const auto rest = k % (kernel_h * kernel_w);
                    const auto kh = rest / kernel_w;
                    const auto kw = rest % kernel_w;
                    auto* dst = output + nip * (kernel_h * kernel_w * output_h * output_w) +
                                kh * (kernel_w * output_h * output_w) + kw * (output_h * output_w);
                    const auto* src = input + nip * (height * width);
                    for (auto y = 0; y < output_h; y++) {
                        const auto iy = y * stride_h + kh;
                        const auto ix = kw;
                        if (stride_w == 1) {
                            memcpy(
                                    dst + (y * output_w),
                                    src + (iy * width + ix),
                                    sizeof(T) * output_w);
                        } else {
                            for (auto x = 0; x < output_w; x++) {
                                memcpy(
                                        dst + (y * output_w + x),
                                        src + (iy * width + ix + x * stride_w),
                                        sizeof(T));
                            }
                        }
                    }
                }
                return;
            }

            // equal padding
            if (pad_l == pad_r && pad_t == pad_b) {
                const int pad_h = pad_t;
                const int pad_w = pad_l;
                const int channel_size = height * width;
                for (int channel = channels; channel--; input += channel_size) {
                    for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
                        for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
                            int input_row = -pad_h + kernel_row * dilation_h;
                            for (int output_rows = output_h; output_rows; output_rows--) {
                                if (!isLess(input_row, height)) {
                                    for (int output_cols = output_w; output_cols; output_cols--) {
                                        *(output++) = 0;
                                    }
                                } else {
                                    int input_col = -pad_w + kernel_col * dilation_w;
                                    for (int output_col = output_w; output_col; output_col--) {
                                        if (isLess(input_col, width)) {
                                            *(output++) = input[input_row * width + input_col];
                                        } else {
                                            *(output++) = 0;
                                        }
                                        input_col += stride_w;
                                    }
                                }
                                input_row += stride_h;
                            }
                        }
                    }
                }
                return;
            }

            // base
            const int dkernel_h = dilation_h * (kernel_h - 1) + 1;
            const int dkernel_w = dilation_w * (kernel_w - 1) + 1;

            int height_col = (height + pad_t + pad_b - dkernel_h) / stride_h + 1;
            int width_col = (width + pad_l + pad_r - dkernel_w) / stride_w + 1;

            int channels_col = channels * kernel_h * kernel_w;
            for (int c = 0; c < channels_col; ++c) {
                int w_offset = c % kernel_w;
                int h_offset = (c / kernel_w) % kernel_h;
                int c_im = c / kernel_h / kernel_w;
                for (int h = 0; h < height_col; ++h) {
                    for (int w = 0; w < width_col; ++w) {
                        int h_pad = h * stride_h - pad_t + h_offset * dilation_h;
                        int w_pad = w * stride_w - pad_l + w_offset * dilation_w;
                        if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width) {
                            output[(c * height_col + h) * width_col + w] = input[(c_im * height + h_pad) * width + w_pad];
                        } else {
                            output[(c * height_col + h) * width_col + w] = 0;
                        }
                    }
                }
            }


        } else if (order == 1) {
            const int dkernel_h = dilation_h * (kernel_h - 1) + 1;
            const int dkernel_w = dilation_w * (kernel_w - 1) + 1;

            int height_col = (height + pad_t + pad_b - dkernel_h) / stride_h + 1;
            int width_col = (width + pad_l + pad_r - dkernel_w) / stride_w + 1;

            int h_pad = -pad_t;
            for (int h = 0; h < height_col; ++h) {
                int w_pad = -pad_l;
                for (int w = 0; w < width_col; ++w) {
                    for (int ih = h_pad; ih < h_pad + dkernel_h; ih += dilation_h) {
                        for (int iw = w_pad; iw < w_pad + dkernel_w; iw += dilation_w) {
                            if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                                memcpy(output, input + (ih * width + iw) * channels, sizeof(T) * channels);
                            } else {
                                memset(output, 0, sizeof(T) * channels);
                            }
                            output += channels;
                        }
                    }
                    w_pad += stride_w;
                }
                h_pad += stride_h;
            }
        } else {
            Logger::Global()->Fatal("Img2Col do not support other image order except NCHW or NHWC \n");
        }

    };


    template <class T, int order>
    inline void Col2Img(const T *input, const int channels, const int height, const int width,
                        const int kernel_h, const int kernel_w, const int dilation_h,
                        const int dilation_w, const int pad_t, const int pad_l,
                        const int pad_b, const int pad_r, const int stride_h, const int stride_w,
                        T *output) {
        memset(output, 0, height * width * channels* sizeof(T));
        if (order == 0) {
            const int output_h = (height + pad_b + pad_t - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
            const int output_w = (width + pad_l + pad_r - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

            if (dilation_h == 1 && dilation_w == 1 && pad_l == 0 && pad_r == 0 &&
                pad_t == 0 && pad_b == 0) {
                for (auto k = 0; k < channels * kernel_h * kernel_w; k++) {
                    const auto nip = k / (kernel_h * kernel_w);
                    const auto rest = k % (kernel_h * kernel_w);
                    const auto kh = rest / kernel_w;
                    const auto kw = rest % kernel_w;
                    const auto* dst = input + nip * (kernel_h * kernel_w * output_h * output_w) +
                                      kh * (kernel_w * output_h * output_w) + kw * (output_h * output_w);
                    auto* src = output + nip * (height * width);
                    for (auto y = 0; y < output_h; y++) {
                        const auto iy = y * stride_h + kh;
                        const auto ix = kw;
                        if (stride_w == 1) {
                            auto offsrc = src + (iy * width + ix);
                            const auto offdst = dst + (y * output_w);
                            for (auto i = 0; i < output_w; ++i) {
                                offsrc[i] += offdst[i];
                            }
                        } else {
                            for (auto x = 0; x < output_w; x++) {
                                auto offsrc = src + (iy * width + ix + x * stride_w);
                                const auto offdst = dst + (y * output_w + x);
                                *offsrc += *offdst;
                            }
                        }
                    }
                }
                return;
            }

            if (pad_l == pad_r && pad_t == pad_b) {
                // From Intel, https://github.com/BVLC/caffe/pull/3536
                const int pad_h = pad_t;
                const int pad_w = pad_l;
                const int channel_size = height * width;
                for (int channel = channels; channel--; output += channel_size) {
                    for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
                        for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
                            int input_row = -pad_h + kernel_row * dilation_h;
                            for (int output_rows = output_h; output_rows; output_rows--) {
                                if (!isLess(input_row, height)) {
                                    input += output_w;
                                } else {
                                    int input_col = -pad_w + kernel_col * dilation_w;
                                    for (int output_col = output_w; output_col; output_col--) {
                                        if (isLess(input_col, width)) {
                                            output[input_row * width + input_col] += *input;
                                        }
                                        input++;
                                        input_col += stride_w;
                                    }
                                }
                                input_row += stride_h;
                            }
                        }
                    }
                }
                return;
            }

            const int dkernel_h = dilation_h * (kernel_h - 1) + 1;
            const int dkernel_w = dilation_w * (kernel_w - 1) + 1;

            int height_col = (height + pad_t + pad_b - dkernel_h) / stride_h + 1;
            int width_col = (width + pad_l + pad_r - dkernel_w) / stride_w + 1;
            int channels_col = channels * kernel_h * kernel_w;
            for (int c = 0; c < channels_col; ++c) {
                int w_offset = c % kernel_w;
                int h_offset = (c / kernel_w) % kernel_h;
                int c_im = c / kernel_h / kernel_w;
                for (int h = 0; h < height_col; ++h) {
                    for (int w = 0; w < width_col; ++w) {
                        int h_pad = h * stride_h - pad_t + h_offset * dilation_h;
                        int w_pad = w * stride_w - pad_l + w_offset * dilation_w;
                        if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width) {
                            output[(c_im * height + h_pad) * width + w_pad] += input[(c * height_col + h) * width_col + w];
                        }
                    }
                }
            }

        } else if (order == 1) {
            const int dkernel_h = dilation_h * (kernel_h - 1) + 1;
            const int dkernel_w = dilation_w * (kernel_w - 1) + 1;

            int height_col = (height + pad_t + pad_b - dkernel_h) / stride_h + 1;
            int width_col = (width + pad_l + pad_r - dkernel_w) / stride_w + 1;
            int h_pad = -pad_t;
            for (int h = 0; h < height_col; ++h) {
                int w_pad = -pad_l;
                for (int w = 0; w < width_col; ++w) {
                    for (int ih = h_pad; ih < h_pad + dkernel_h; ih += dilation_h) {
                        for (int iw = w_pad; iw < w_pad + dkernel_w; iw += dilation_w) {
                            if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                                auto* data_im_patch = output + (ih * width + iw) * channels;
                                Add<T>(channels, data_im_patch, input, data_im_patch);
                            }
                            input += channels;
                        }
                    }
                    w_pad += stride_w;
                }
                h_pad += stride_h;
            }
        } else {
            Logger::Global()->Fatal("Col2Img do not support other image order except NCHW or NHWC \n");
        }
    };



    template <class T>
    inline void Img2ColNd(const T *input, const int *imageShape, const int *dataShape,
                          const int * kernel, const int *stride, const int * dilation,
                          const int * padding, const int N, T *output, bool col2img = false) {
        int kernel_size = 1;
        for (int i = 0; i < N; ++i) {
            kernel_size *= kernel[i];
        }
        const int channels_col = dataShape[0];
        std::vector<int> d_offset(N, 0);
        std::vector<int> d_iter(N, 0);
        for (int c_col = 0; c_col < channels_col; ++c_col) {
            int offset = c_col;
            for (int d_i = N - 1; d_i >= 0; --d_i) {
                if (d_i < N - 1) {
                    offset /= kernel[d_i + 1];
                }
                d_offset[d_i] = offset % kernel[d_i];
            }
            for (bool incremented = true; incremented;) {
                int index_col = c_col;
                int index_im = c_col / kernel_size;
                bool is_padding = false;
                for (int d_i = 0; d_i < N; ++d_i) {
                    const int d = d_iter[d_i];
                    const int d_im = d * stride[d_i] - padding[d_i] + d_offset[d_i] * dilation[d_i];
                    is_padding |= d_im < 0 || d_im >= imageShape[d_i + 1];
                    index_col *= dataShape[d_i + 1];
                    index_col += d;
                    index_im *= imageShape[d_i + 1];
                    index_im += d_im;
                }
                if (!col2img) {
                    if (is_padding) {
                        output[index_col] = 0;
                    } else {
                        output[index_col] = input[index_im];
                    }
                } else if (!is_padding) { // col2im
                    output[index_im] += input[index_col];
                }
                incremented = false;
                for (int d_i = N - 1; d_i >= 0; --d_i) {
                    const int d_max = dataShape[d_i + 1];
                    if (d_iter[d_i] < d_max) {
                        Logger::Global()->Fatal("Img2ColNd d_iter[%d] less then d_max\n", d_i);
                    }
                    if (d_iter[d_i] == d_max - 1) {
                        d_iter[d_i] = 0;
                    } else { // d_iter[d_i] < d_max - 1
                        ++d_iter[d_i];
                        incremented = true;
                        break;
                    }
                }
            }
        }
    };



    template <class T>
    inline void Col2ImgNd(const T *input, const int *imageShape, const int *dataShape,
                          const int * kernel, const int *stride, const int * dilation,
                          const int * padding, const int N, T *output) {
        int imageSize = 1;
        for (int i = 0; i < N; ++i) {
            imageSize *= imageShape[i];
        }
        memset(output, 0, sizeof(T) * imageSize);
        Img2ColNd(input, imageShape, dataShape, kernel, stride, dilation, padding, N, output, true);
    }

    template<class T>
    inline void img2col(const T *input, const int input_channels, const int input_width, const int input_height,
                        const int stride_width, const int stride_height, const int padding_width,
                        const int padding_height, const int filter_width, const int filter_height,
                        const int dilation_width, const int dilation_height, T *output) {
        const int output_width =
                (input_width + 2 * padding_width - (dilation_width * (filter_width - 1) + 1)) / stride_width + 1;
        const int output_height =
                (input_height + 2 * padding_height - (dilation_height * (filter_height - 1) + 1)) / stride_height + 1;
        const int col_channels = input_channels * filter_width * filter_height;
#ifdef USE_MP
#pragma omp parallel for
#endif
        for (int c = 0; c < col_channels; ++c) {
            int w_offset = c % filter_width;
            int h_offset = (c / filter_width) % filter_height;
            int c_im = c / filter_width / filter_height;
            for (int h = 0; h < output_height; ++h) {
                for (int w = 0; w < output_width; ++w) {
                    int imRowIdx = h * stride_height + h_offset * dilation_height;
                    int imColIdx = w * stride_width + w_offset * dilation_width;
                    if ((imRowIdx - padding_height) < 0 ||
                        (imRowIdx - padding_height) >= input_height ||
                        (imColIdx - padding_width) < 0 ||
                        (imColIdx - padding_width) >= input_width) {
                        output[(c * output_height + h) * output_width + w] = T(0);
                    } else {
                        imRowIdx += c_im * input_height - padding_height;
                        imColIdx -= padding_width;
                        output[(c * output_height + h) * output_width + w] =
                                input[imRowIdx * input_width + imColIdx];
                    }
                }
            }
        }
    }


    template<class T>
    inline void col2img(T *input, const int input_channels, const int input_width, const int input_height,
                        const int stride_width, const int stride_height, const int padding_width,
                        const int padding_height, const int filter_width, const int filter_height,
                        const int dilation_width, const int dilation_height, const T *output) {
        const int output_width =
                (input_width + 2 * padding_width - (dilation_width * (filter_width - 1) + 1)) / stride_width + 1;
        const int output_height =
                (input_height + 2 * padding_height - (dilation_height * (filter_height - 1) + 1)) / stride_height + 1;
        const int col_channels = input_channels * filter_width * filter_height;
#ifdef USE_MP
#pragma omp parallel for
#endif
        for (int c = 0; c < col_channels; ++c) {
            int w_offset = c % filter_width;
            int h_offset = (c / filter_width) % filter_height;
            int c_im = c / filter_width / filter_height;
            for (int h = 0; h < output_height; ++h) {
                for (int w = 0; w < output_width; ++w) {
                    int imRowIdx = h * stride_height + h_offset * dilation_height;
                    int imColIdx = w * stride_width + w_offset * dilation_width;
                    imRowIdx -= padding_height;
                    imColIdx -= padding_width;
                    if (imRowIdx >= 0 && imRowIdx < input_height &&
                        imColIdx >= 0 && imColIdx < input_width) {
                        int input_idx = (imRowIdx + c_im * input_height) * input_width + imColIdx;
                        int output_idx = (c * output_height + h) * output_width + w;
                        input[input_idx] += output[output_idx];
                    }
                }
            }
        }
    }


    template<class T>
    inline void NaiveConv(const T *input, const int batch_size, const int input_channels,
                          const int input_width, const int input_height,
                          const int stride_width, const int stride_height,
                          const int padding_width, const int padding_height,
                          const int filter_width, const int filter_height,
                          const int dilation_width, const int dilation_height,
                          const int output_channels, const T *filter, T *output) {

        const int output_width =
                (input_width + 2 * padding_width - (dilation_width * (filter_width - 1) + 1)) / stride_width + 1;
        const int output_height =
                (input_height + 2 * padding_height - (dilation_height * (filter_height - 1) + 1)) / stride_height + 1;
#ifdef USE_MP
#pragma omp parallel for
#endif
        for (int batch = 0; batch < batch_size; ++batch) {
            for (int out_channel = 0; out_channel <output_channels ; ++out_channel) {
                for (int out_h = 0; out_h < output_height; ++out_h) {
                    for (int out_w = 0; out_w < output_width; ++out_w) {
                        const int inStartH = (out_h * stride_height) - padding_height;
                        const int inStartW = (out_w * stride_width) - padding_width;
                        T outValue = (T)0;
                        for (int in_channel = 0; in_channel < input_channels; ++in_channel) {
                            for (int filter_h = 0; filter_h < filter_height; ++filter_h) {
                                for (int filter_w = 0; filter_w < filter_width; ++filter_w) {
                                    T inValue;
                                    const int inH = inStartH + filter_h;
                                    const int inW = inStartW + filter_w;
                                    if ((inH >= 0 && inH < input_height) &&
                                        (inW >= 0 && inW < input_width)) {
                                        int offsetInput = batch * input_channels * input_height * input_width +
                                                        in_channel * input_height * input_width + inH * input_width + inW;
                                        inValue = input[offsetInput];
                                    } else {
                                        inValue = (T)0;
                                    }
                                    int offsetFilter = out_channel * input_channels * filter_height * filter_width +
                                                    in_channel * filter_height * filter_width + filter_h * filter_width + filter_w;
                                    T filterValue = filter[offsetFilter];
                                    outValue += (inValue * filterValue);
                                }
                            }

                        }
                        int offset = batch * output_channels * output_height * output_width +
                                        out_channel * output_height * output_width + out_h * output_width + out_w;
                        output[offset] = outValue;
                    }
                }
            }
        }
    }




    template <class T>
    inline void pooling2D(const T *input, const int batch_size, const int channel,
                          const int input_width, const int input_height,
                          const int output_width, const int output_height,
                          const int stride_width, const int stride_height,
                          const int padding_width, const int padding_height,
                          const int filter_width, const int filter_height,
                          const int dilation_width, const int dilation_height,
                          T *output, int type = 0, T *mask = nullptr) {

        const int input_stride = input_height * input_width;
        const int output_stride = output_height * output_width;
#ifdef USE_MP
#pragma omp parallel for
#endif
        for (int i = 0; i < batch_size; ++i) {
            for (int c = 0; c < channel; ++c) {
                for (int ph = 0; ph < output_height; ++ph) {

                    int hstart = ph * stride_height - padding_height;
                    int hend = std::min(hstart + filter_height, input_height);
                    hstart = std::max(hstart, 0);
                    for (int pw = 0; pw < output_width; ++pw) {

                        int wstart = pw * stride_width - padding_width;
                        int wend = std::min(wstart + filter_width, input_width);
                        wstart = std::max(wstart, 0);
                        T ele;
                        if (type == 0) {
                            ele = input[hstart * input_width + wstart];
                            int index = hstart * input_width + wstart;
                            for (int h = hstart; h < hend; ++h) {
                                for (int w = wstart; w < wend; ++w) {
                                    if (ele < input[h * input_width + w]) {
                                        ele = input[h * input_width + w];
                                        index = h * input_width + w;
                                    }
                                }
                            }
                            output[ph * output_width + pw] = ele;
                            if (mask != nullptr) {
                                mask[ph * output_width + pw] = T(index);
                            }
                        } else if (type == 1) {
                            ele = T(0);
                            for (int h = hstart; h < hend; ++h) {
                                for (int w = wstart; w < wend; ++w) {
                                    ele += input[h * input_width + w];
                                }
                            }
                            output[ph * output_width + pw] = ele / (hend * wend);
                        } else {
                            Logger::Global()->Fatal("not Implementation Pooling2D with other PoolType");
                        }


                    }
                }

                input += input_stride;
                output += output_stride;
                if (mask != nullptr) {
                    mask += output_stride;
                }
            }
        }
    }

    template <class T>
    inline void NHWC2NCHW(const T * input,
                          const int num,
                          const int inH,
                          const int inW,
                          const int inC,
                          T *output) {
#ifdef USE_MP
#pragma omp parallel for
#endif
        for (int n = 0; n < num; ++n) {
            for (int h = 0; h < inH; ++h) {
                for (int w = 0; w < inW; ++w) {
                    for (int c = 0; c < inC; ++c) {
                        output[((n * inC + c) * inH + h) * inW + w] = *(input++);
                    }
                }
            }
        }
    }


    template <class T>
    inline void NCHW2NHWC(const T * input,
                          const int num,
                          const int inC,
                          const int inH,
                          const int inW,
                          T *output) {
#ifdef USE_MP
#pragma omp parallel for
#endif
        for (int n = 0; n < num; ++n) {
            for (int c = 0; c < inC; ++c) {
                for (int h = 0; h < inH; ++h) {
                    for (int w = 0; w < inW; ++w) {
                        output[((n * inH + h) * inW + w) * inC + c] = *(input++);
                    }
                }
            }
        }
    }

    template <class I, class R>
    inline std::vector<R> Map(std::function<R(I)> fun, const std::vector<I>& vec) {
        std::vector<R> res;
        res.reserve(vec.size());
#ifdef USE_MP
#pragma omp parallel for
#endif
        for (auto& i : vec) {
            res.push_back(fun(i));
        }
        return res;
    };


    template <typename R, typename I>
    inline std::vector<R> Map(std::function<R(I)> fun, std::vector<I>&& vec) {
        return Map<R, I>(fun, vec);
    }

    template <typename I>
    inline I Reduce(std::function<I(const I&, const I&)> func, I initVal, const std::vector<I>& vec) {
        I res = initVal;
#ifdef USE_MP
#pragma omp parallel for
#endif
        for (int i = 0; i < vec.size(); ++i) {
            res = func(res, vec.at(i));
        }
        return res;
    }

    template <typename I>
    inline I Reduce(std::function<I(const I&, const I&)> func, const std::vector<I>& vec) {
        const std::vector<I> v(vec.begin() + 1, vec.end());
        return Reduce(func, vec.at(0), v);
    }

    template <typename I>
    inline I Reduce(std::function<I(I&&, I&&)> func, I&& initVal, std::vector<I>&& vec) {
#ifdef USE_MP
#pragma omp parallel for
#endif
        I res = std::move(initVal);
        for (int i = 0; i < vec.size(); ++i) {
            res = func(std::move(res), std::move(vec.at(i)));
        }
        return res;
    }

    template <typename I>
    I Reduce(std::function<I(I&&, I&&)> func, std::vector<I>&& vec) {
#ifdef USE_MP
#pragma omp parallel for
#endif
        I res = std::move(vec.at(0));
        for (int i = 1; i < vec.size(); ++i) {
            res = func(std::move(res), std::move(vec.at(i)));
        }
        return res;
    }


    template <typename R, typename I>
    inline R MapReduce(std::function<R(R, I, bool)> func, const std::vector<I>& vec) {
#ifdef USE_MP
#pragma omp parallel for
#endif
        R res = func(R(), vec.at(0), true);
        for (int i = 1; i < vec.size(); ++i) {
            res = func(res, vec.at(i), false);
        }
        return res;
    }

    template< class T>
    inline std::vector<T> Filter(std::function<bool(const T)> func, const std::vector<T> &input) {
        std::vector<T> res;
        res.reserve(input.size());
        for (const auto &i : input) {
            if (func(i)) {
                res.push_back(i);
            }
        }
        res.shrink_to_fit();
        return res;
    }
}

#endif //MATRIX_MATH_H
