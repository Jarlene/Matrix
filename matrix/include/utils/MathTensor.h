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
    void Copy(const Tensor<T> &input, Tensor<T> &out) {
        if (input.GetShape() == out.GetShape()) {
            CPUCopy<T>(input.Size(), input.Data(), 1, out.MutableData(), 1);
        } else  {
            Shape in = input.GetShape();
            Shape ou = out.GetShape();
            if (in.isMatrix() && ou.isVector() && in[0] == ou[0]) {
                SumCopy<T>(in.Size(), input.Data(), ou.Size(), out.MutableData());
            }
        }

    }

    template <class T>
    void ApplyNode(Tensor<T> &out,  Tensor<T> &grad, T learning_rate) {
        CPUAxpy(out.Size(), learning_rate, grad.Data(), 1, out.MutableData(), 1);
    }

    template <class T>
    void applyMomentum(Tensor<T> &base, Tensor<T> &grad, T alpha, T beta) {
        CPUAxpby(base.Size(), alpha, grad.Data(), 1, beta, base.MutableData(), 1);
    }

    template <class T>
    void MatrixMul(const Tensor<T> &a, const bool ATran,  const Tensor<T> &b, const bool BTran, Tensor<T> &c, T beta = T(0)) {
        assert(ATran ? a.GetShape()[1] : a.GetShape()[0] == c.GetShape()[0]);
        assert(ATran ? a.GetShape()[0] : a.GetShape()[1] == BTran ? b.GetShape()[1] : b.GetShape()[0]);
        assert(BTran ? b.GetShape()[0] : b.GetShape()[1] == c.GetShape()[1]);

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
        if (a.GetShape() == b.GetShape()) {
            int size = a.Size();
            Add<T>(size, a.Data(), b.Data(), c.MutableData());
        } else if (a.GetShape()[0] == b.GetShape()[0] && b.Rank() == 1) {
            Add<T>(a.GetShape()[0], a.GetShape()[1], a.Data(), b.Data(), c.MutableData());
        } else if (a.GetShape()[1] == b.GetShape()[0] && b.Rank() == 1) {
            int size = b.GetShape()[0];
            int unit = a.GetShape()[0];
            for (int i = 0; i < unit; ++i) {
                Add<T>(size, a.Data(i * size), b.Data(), c.MutableData(i * size));
            }
        } else {
            Logger::Global()->Fatal("Add not support add method now.");
        }

    }

    template <class T>
    void Scale(Tensor<T> &out, T val) {
        Scale<T>(out.Size(), out.MutableData(), val);
    }

    template <class T>
    void Sub(const Tensor<T> &a, const Tensor<T> &b,  Tensor<T> &c) {
        assert(a.GetShape() == b.GetShape());
        assert(a.GetShape() == c.GetShape());
        int size = a.Size();
        Sub<T>(size, a.Data(), b.Data(), c.MutableData());
    }

    template <class T>
    void Div(const Tensor<T> &a, const Tensor<T> &b,  Tensor<T> &c) {

    }


    template <class T>
    void Sigmoid(const Tensor<T> &input, Tensor<T> &out) {
        assert(input.GetShape() == out.GetShape());
        Sigmoid<T>(input.Size(), input.Data(), out.MutableData());
    }

    template <class T>
    void SigmoidGrad(const Tensor<T> &input, const Tensor<T> &pre, Tensor<T> &out) {
        SigmoidGrad<T>(input.Size(), input.Data(), pre.Data(), out.MutableData());
    }

    template <class T>
    void Tanh(const Tensor<T> &a, Tensor<T> &b) {
        assert(a.GetShape() == b.GetShape());
        int size = a.Size();
        Tanh<T>(size, a.Data(), b.MutableData());
    }

    template <class T>
    void TanhGrad(const Tensor<T> &input, const Tensor<T> &pre,  Tensor<T> &out) {
        TanhGrad<T>(input.Size(), input.Data(), pre.Data(), out.MutableData());
    }

    template <class T>
    void Relu(const Tensor<T> &input, Tensor<T> &out) {
        assert(input.GetShape() == out.GetShape());
        Relu<T>(input.Size(), input.Data(), out.MutableData());
    }

    template <class T>
    void ReluGrad(const Tensor<T> &input, const Tensor<T> &pre,  Tensor<T> &out) {
        ReluGrad<T>(input.Size(), input.Data(), pre.Data(), out.MutableData());
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
    void CrossEntropy(const Tensor<T> &data, const Tensor<T> &label, Tensor<T> &out) {
        CrossEntropy<T>(data.Size(), data.Data(), label.Size(), label.Data(), out.MutableData());
    }

    template <class T>
    void CrossEntropyGrad(const Tensor<T> &data, const Tensor<T> &label, Tensor<T> &out) {
        CrossEntropyGrad(data.Size(), data.Data(), label.Size(), label.Data(), out.MutableData());
    }

    template <class T>
    void RMSLoss(const Tensor<T> &data, const Tensor<T> &label, Tensor<T> &out) {
        RMSLoss<T>(data.Size(), data.Data(), label.Size(), label.Data(), out.MutableData());
    }

    template <class T>
    void RMSLossGrad(const Tensor<T> &data, const Tensor<T> &label, Tensor<T> &out) {
        RMSLossGrad<T>(data.Size(), data.Data(), label.Size(), label.Data(), out.MutableData());
    }

    template <class T>
    void Softmax(const Tensor<T> &data, Tensor<T> &out) {
        int batch_size = data.GetShape()[0];
        int class_num = data.GetShape()[1];
        for (int i = 0; i < batch_size; ++i) {
            Softmax<T>(class_num,  data.Data(i * class_num), out.MutableData(i * class_num));
        }

    }

    template <class T>
    void SoftmaxGrad(const Tensor<T> &data, const Tensor<T> &pre_grad, Tensor<T> &out) {
        SoftmaxGrad(data.GetShape()[0], data.GetShape()[1], data.Data(), pre_grad.Data(), out.MutableData());
    }


    template <class T>
    void SoftmaxCrossEntropy(const Tensor<T> &data, const Tensor<T> &label, Tensor<T> &out) {
        SoftmaxCrossEntropy(data.Size(), data.Data(), label.Size(), label.Data(), out.MutableData());
    }

    template <class T>
    void SoftmaxCrossEntropyGrad(const Tensor<T> &data, const Tensor<T> &label, Tensor<T> &out) {
        SoftmaxCrossEntropyGrad(data.Size(), data.Data(), label.Size(), label.Data(), out.MutableData());
    }


//    template <class T>
//    inline void Img2Col(const Tensor<T> &input, Shape &kernel, Shape &stride,
//                        Shape &padding, Shape &dilate,  Tensor<T> &output, ImageOrder order = NCHW) {
//
//        if (kernel.Rank() == 2) {
//            Shape in = input.GetShape();
//            assert(in.Rank() == 4);
//            switch (order) {
//                case NCHW:
//                    Img2Col<T, 0>(input.Data(), in[1], in[2], in[3],
//                                      kernel[0], kernel[1],
//                                      dilate[0], dilate[1],
//                                      padding[0],padding[1],
//                                      padding[0],padding[1],
//                                      stride[0],stride[1],
//                                      output.MutableData());
//                    break;
//                case NHWC:
//                    Img2Col<T, 1>(input.Data(), in[1], in[2], in[3],
//                                  kernel[0], kernel[1],
//                                  dilate[0], dilate[1],
//                                  padding[0],padding[1],
//                                  padding[0],padding[1],
//                                  stride[0],stride[1],
//                                  output.MutableData());
//                    break;
//                default:
//                    Logger::Global()->Fatal("Col2Img does not support");
//                    break;
//            }
//
//        } else {
//            Shape colShape;
//            int c = 0;
//            switch (order) {
//                case NCHW:
//                    c = input.GetShape()[1];
//                    colShape.Append(c*kernel.Size());
//                    for (int i = 1; i < kernel.Rank(); ++i) {
//                        colShape.Append(output.GetShape()[i+1]);
//                    }
//                    break;
//                case NHWC:
//                    c = input.GetShape()[3];
//                    colShape.Append(c*kernel.Size());
//                    for (int i = 1; i < kernel.Rank(); ++i) {
//                        colShape.Append(output.GetShape()[i+1]);
//                    }
//                    break;
//                default:
//                    Logger::Global()->Fatal("Col2Img does not support");
//                    break;
//            }
//            Img2ColNd<T>(input.Data(), input.GetShape().Array(), colShape.Array(),
//                         kernel.Array(), stride.Array(),
//                         dilate.Array(), padding.Array(), kernel.Rank(),
//                         output.MutableData());
//        }
//
//    }
//
//
//    template <class T>
//    inline void Col2Img(const Tensor<T> & input, Shape &kernel, Shape &stride,
//                        Shape &padding, Shape &dilate, Tensor<T> &output, ImageOrder order = NCHW) {
//        Shape in = input.GetShape();
//        if (kernel.Rank() == 2) {
//            assert(in.Rank() == 4);
//            switch (order) {
//                case NCHW:
//                    Col2Img<T, 0>(input.Data(), in[1], in[2], in[3],
//                                    kernel[0], kernel[1],
//                                    dilate[0], dilate[1],
//                                    padding[0],padding[1],
//                                    padding[0],padding[1],
//                                    stride[0],stride[1],
//                                    output.MutableData());
//                    break;
//                case NHWC:
//                    Col2Img<T, 1>(input.Data(), in[1], in[2], in[3],
//                                  kernel[0], kernel[1],
//                                  dilate[0], dilate[1],
//                                  padding[0],padding[1],
//                                  padding[0],padding[1],
//                                  stride[0],stride[1],
//                                  output.MutableData());
//                    break;
//                default:
//                    Logger::Global()->Fatal("Col2Img does not support");
//                    break;
//            }
//
//        } else {
//            Shape colShape;
//            int c = 0;
//            switch (order) {
//                case NCHW:
//                    c = in[1];
//                    colShape.Append(c*kernel.Size());
//                    for (int i = 1; i < kernel.Rank(); ++i) {
//                        colShape.Append(output.GetShape()[i+1]);
//                    }
//                    break;
//                case NHWC:
//                    c = in[3];
//                    colShape.Append(c*kernel.Size());
//                    for (int i = 1; i < kernel.Rank(); ++i) {
//                        colShape.Append(output.GetShape()[i+1]);
//                    }
//                    break;
//                default:
//                    Logger::Global()->Fatal("Col2Img does not support");
//                    break;
//            }
//
//            Col2ImgNd<T>(input.Data(), in.Array(), colShape.Array(),
//                         kernel.Array(), stride.Array(),
//                         dilate.Array(), padding.Array(), kernel.Rank(),
//                         output.MutableData());
//        }
//    }
//
//
//
//
//    template <class T>
//    inline void Img2ColND(const Tensor<T> &input, Shape &colShape, Shape &kernel, Shape &stride,
//                          Shape &padding, Shape &dilate, Tensor<T> &output, ImageOrder order = NCHW) {
//
//        Col2ImgNd<T>(input.Data(), input.GetShape().Array(),
//                     colShape.Array(), kernel.Array(),stride.Array(),
//                     dilate.Array(), padding.Array(), kernel.Rank(), output.MutableData());
//
//    }

    template <class T>
    inline void Img2Col(const T* input, const Shape &input_shape, const Shape &kernel, const Shape &stride,
                        const Shape &padding, const Shape &dilate, T *output, const ImageOrder &order = NCHW) {
        int input_channels = 1;
        int input_width = 1;
        int input_height = 1;
        switch(order) {
            case NCHW:
                input_channels = input_shape[1];
                input_height = input_shape[2];
                input_width = input_shape[3];
                break;
            case NHWC:
                input_channels = input_shape[3];
                input_height = input_shape[1];
                input_width = input_shape[2];
                break;
            default:
                Logger::Global()->Fatal("Img2Col does not support\n");
                break;
        }
        int stride_width = stride[0];
        int stride_height = stride[1];
        int padding_width = padding[0];
        int padding_height = padding[1];
        int dilate_width = dilate[0];
        int dilate_height = dilate[1];
        int filter_width = kernel[2];
        int filter_height = kernel[3];
        img2col<T>(input, input_channels,
                   input_width, input_height,
                   stride_width,stride_height,
                   padding_width,padding_height,
                   filter_width,filter_height,
                   dilate_width,dilate_height,
                   output);
    }


    template <class T>
    inline void Col2Img(const T* input, const Shape &input_shape, const Shape &kernel, const Shape &stride,
                        const Shape &padding, const Shape &dilate, T *output, const ImageOrder &order = NCHW) {
        int input_channels = 1;
        int input_width = 1;
        int input_height = 1;
        switch(order) {
            case NCHW:
                input_channels = input_shape[1];
                input_height = input_shape[2];
                input_width = input_shape[3];
                break;
            case NHWC:
                input_channels = input_shape[3];
                input_height = input_shape[1];
                input_width = input_shape[2];
                break;
            default:
                Logger::Global()->Fatal("Col2Img does not support\n");
                break;
        }
        int stride_width = stride[0];
        int stride_height = stride[1];
        int padding_width = padding[0];
        int padding_height = padding[1];
        int dilate_width = dilate[0];
        int dilate_height = dilate[1];
        int filter_width = kernel[2];
        int filter_height = kernel[3];
        col2img<T>(output, input_channels,
                   input_width,input_height,
                   stride_width,stride_height,
                   padding_width,padding_height,
                   filter_width,filter_height,
                   dilate_width,dilate_height,
                   input);
    }
}

#endif //MATRIX_MATHTENSOR_H
