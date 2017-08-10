//
// Created by Jarlene on 2017/8/5.
//

#ifndef MATRIX_MATHTENSOR_H
#define MATRIX_MATHTENSOR_H

#include <cassert>
#include "matrix/include/api/MatrixType.h"
#include "matrix/include/base/Tensor.h"
#include "Math.h"
#include "Logger.h"

namespace matrix {


    template <class T>
    void MatrixMul(const Tensor<T> &a, const bool ATran,  const Tensor<T> &b, const bool BTran, Tensor<T> &c, T beta = T(0)) {
        assert(a.GetShape()[0] == c.GetShape()[0]);
        assert(a.GetShape()[1] == b.GetShape()[0]);
        assert(b.GetShape()[1] == c.GetShape()[1]);

        int m = ATran ? a.GetShape()[1] : a.GetShape()[0];
        int n = BTran ? b.GetShape()[0] : b.GetShape()[1];
        int k = ATran ? a.GetShape()[0] : a.GetShape()[1];
        CPUGemm<T>(ATran ? Trans : NoTrans, BTran ? Trans : NoTrans, m, n, k,
                   T(1), a.Data(), b.Data(), beta, c.MutableData());
    }


    template <class T>
    void MatrixMulVector(const Tensor<T> &a, const bool ATran, const Tensor<T> &b, Tensor<T> &c, T beta = T(0)) {
        assert(a.Rank() == 2);
        assert((b.Rank() == 1 || (b.Rank() == 2 && b.GetShape()[1] == 1)));
        assert((ATran? a.GetShape()[0] == b.GetShape()[0] : a.GetShape()[1] == b.GetShape()[0]));
        int m = ATran? a.GetShape()[1] : a.GetShape()[0];
        int n = ATran? a.GetShape()[0] : a.GetShape()[1];
        CPUGemv<T>(ATran? Trans: NoTrans, m, n, T(1), a.Data(), b.Data(), beta, c.MutableData());
    }


    template <class T>
    void Add(const Tensor<T> &a, const Tensor<T> &b, Tensor<T> &c) {
        assert(a.GetShape() == c.GetShape());
        int size = a.Size();
        if (a.GetShape() == b.GetShape()) {
            Add<T>(size, a.Data(), b.Data(), c.MutableData());
        } else if (a.GetShape()[0] == b.GetShape()[0] && b.Rank() == 1) {
            Add<T>(a.GetShape()[0], a.GetShape()[1], a.Data(), b.Data(), c.MutableData());
        } else {
            Logger::Global()->Fatal("not support add method now \n");
        }

    }



    template <class T>
    void Sub(const Tensor<T> &a, const Tensor<T> &b,  Tensor<T> &c) {
        assert(a.GetShape() == b.GetShape());
        assert(a.GetShape() == c.GetShape());
        int size = a.Size();
        Sub<T>(size, a.Data(), b.Data(), c.MutableData());
    }



    template <class T>
    void Tanh(const Tensor<T> &a, Tensor<T> &b) {
        assert(a.GetShape() == b.GetShape());
        int size = a.Size();
        Tanh<T>(size, a.Data(), b.MutableData());
    }

    template <class T>
    void GradTanh(const Tensor<T> &a, const Tensor<T> &b,  Tensor<T> &c) {
        TanhGrad<T>(a.Size(), a.Data(), b.Data(), c.MutableData());
    }

    template <class T>
    void Value(Tensor<T> &a, T value) {
        Value<T>(a.Size(), a.MutableData(), value);
    }

    template <class T>
    void Random(Tensor<T> &a, T mu, T sigma) {
        Random<T>(a.Size(), a.MutableData(), mu, sigma);
    }


    template <class T>
    inline void Img2Col(const Tensor<T> &input, Shape &in, Shape &kernel, Shape &stride,
                        Shape &padding, Shape &dilate, Shape &out, Tensor<T> &output, ImageOrder &order) {

        if (kernel.Rank() == 2) {
            assert(in.Rank() == 4);
            Img2Col<T, order>(input.Data(), in[1], in[2], in[3],
                              kernel[0], kernel[1],
                              dilate[0], dilate[1],
                              padding[0],padding[1],
                              padding[0],padding[1],
                              stride[0],stride[1],
                              output.MutableData());
        } else {
            Img2ColNd<T>(input.Data(), in.Array(), out.Array(),
                         kernel.Array(), stride.Array(),
                         dilate.Array(), padding.Array(), kernel.Rank(),
                         output.MutableData());
        }

    }


    template <class T>
    inline void Col2Img(const Tensor<T> & input, Shape &in, Shape &kernel, Shape &stride,
                        Shape &padding, Shape &dilate, Shape &out, Tensor<T> &output, ImageOrder &order) {
        if (kernel.Rank() == 2) {
            assert(in.Rank() == 4);
            Col2Img<T, order>(input.Data(), in[1], in[2], in[3],
                              kernel[0], kernel[1],
                              dilate[0], dilate[1],
                              padding[0],padding[1],
                              padding[0],padding[1],
                              stride[0],stride[1],
                              output.MutableData());
        } else {
            Col2ImgNd<T>(input.Data(), in.Array(), out.Array(),
                         kernel.Array(), stride.Array(),
                         dilate.Array(), padding.Array(), kernel.Rank(),
                         output.MutableData());
        }
    }

}

#endif //MATRIX_MATHTENSOR_H
