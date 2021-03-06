//
// Created by Jarlene on 2017/7/24.
//
#include <gtest/gtest.h>
#include "include/Test.h"
#include <matrix/include/base/Tensor.h>
#include <matrix/include/utils/MathTensor.h>
#include <matrix/include/op/ReduceOp.h>

namespace matrix {

    namespace Test {

        class MathTest : public ::testing::Test {

        };

        TEST_F(MathTest, MatrixMul) {
            float a[] = {1, 2, 3, 4, 5, 6};
            float b[] = {2, 3, 4, 5, 6, 6};
            float c[] = {0, 0, 0, 0};

            float res[] = {28, 31, 64, 73};

            auto at = TensorN(a, 2, 3);
            auto bt = TensorN(b, 3, 2);
            auto ct = TensorN(c, 2, 2);

            MatrixMul(at, false, bt, false, ct);
            checkArrayEqual(c, res, 4);
        }

        TEST_F(MathTest, MatrixVector) {
            float a[] = {1, 2, 3, 4, 5, 6};
            float b[] = {2, 3};
            float c[] = {0, 0, 0, 0, 0, 0};
            float res[] = {8, 18, 28};

            auto at = TensorN(a, 3, 2);
            auto bt = TensorN(b, 2);
            auto ct = TensorN(c, 3);

            MatrixMulVector(at, false, bt, ct);
            checkArrayEqual(c, res, 3);
        }


        TEST_F(MathTest, Add) {
            float a[] = {1, 2, 3, 4, 5, 6};
            float b[] = {2, 3, 4, 5, 6, 6};
            float c[] = {0, 0, 0, 0, 0, 0};
            float res[] = {3, 5, 7, 9, 11, 12};

            auto at = TensorN(a, 2, 3);
            auto bt = TensorN(b, 2, 3);
            auto ct = TensorN(c, 2, 3);

            Add(at, bt, ct);
            checkArrayEqual(c, res, 6);
        }


        TEST_F(MathTest, im2col) {
            float a[] = {0, 1, 2,
                         3, 4, 5};

            float b[8] = {0};

            float target[8] = {0, 1,
                               1, 2,
                               3, 4,
                               4, 5};

            Shape in = ShapeN(1, 1, 2, 3);
            Shape kernel = ShapeN(1, 1, 2, 2);
            Shape padding = ShapeN(0, 0);
            Shape stride = ShapeN(1, 1);
            Shape dilation = ShapeN(1, 1);

            PrintMat(a, 2, 3, "orig_data");
            Img2Col(a, in, kernel, stride, padding, dilation, b);
            PrintMat(b, 4, 2, "im2col");
            checkArrayEqual<float>(b, target, 8);
        }

        TEST_F(MathTest, col2im) {
            float a[] = {0, 1, 1, 2,
                         3, 4, 4, 5};

            float b[6] = {0};

            float target[6] = {0, 2, 2,
                               3, 8, 5};

            Shape in = ShapeN(1, 1, 2, 3);
            Shape kernel = ShapeN(1, 1, 2, 2);
            Shape padding = ShapeN(0, 0);
            Shape stride = ShapeN(1, 1);
            Shape dilation = ShapeN(1, 1);

            PrintMat(a, 2, 4, "orig_data");
            Col2Img(a, in, kernel, stride, padding, dilation, b);
            PrintMat(b, 2, 3, "col2im");
            checkArrayEqual<float>(b, target, 6);

        }


        TEST_F(MathTest, NaiveConv) {
            float a[] = {1, 1, 1, 0, 0,
                         0, 1, 1, 1, 0,
                         0, 0, 1, 1, 1,
                         0, 0, 1, 1, 0,
                         0, 1, 1, 0, 0};

            float kernel[] = {1, 1, 1,
                              1, 1, 0,
                              1, 0, 0};

            float b[9];

            float c[] = {4, 4, 4,
                         2, 4, 5,
                         1, 4, 6};

            NaiveConv(a, 1, 1, 5, 5, 1, 1, 0, 0, 3, 3, 1, 1, 1, kernel, b);
            checkArrayEqual<float>(b, c, 9);
        }


        TEST_F(MathTest, PoolMax) {
            float a[] = {1, 3, 1, 5,
                         4, 2, 6, 2,
                         7, 1, 3, 5,
                         1, 3, 10, 2};


            float b[9] = {0};
            float c[9] = {0};
            PrintMat(a, 4, 4, "orig_data");
            pooling2D(a, 1, 1, 4, 4, 3, 3, 1, 1, 0, 0, 2, 2, 1, 1, b, kMax, c);
            PrintMat(b, 1, 9, "out");
            PrintMat(c, 1, 9, "index");

            float target[] = {4, 6, 6, 7, 6, 6, 7, 10, 10};
            float indexTarget[] = {4, 6, 6, 8, 6, 6, 8, 14, 14};
            checkArrayEqual<float>(b, target, 9);
            checkArrayEqual<float>(c, indexTarget, 9);
        }

        TEST_F(MathTest, Softmax) {
            float a[] = {2, 1, 3, 5,
                         3, 2, 6, 1,
                         8, 2, 4, 6,
                         5, 9, 3, 2};
            float b[16] = {0};
            float target[16] = {0.0414, 0.0152, 0.1125, 0.8310,
                                0.0463, 0.0170, 0.9304, 0.0063,
                                0.8650, 0.0021, 0.0158, 0.1171,
                                0.0179, 0.9788, 0.0024, 0.0009};
            for (int i = 0; i < 4; ++i) {
                Softmax(4, a + 4 * i, b + 4 * i);
            }
            checkArrayEqual(b, target, 16);
            PrintMat(b, 4, 4, "softmax_out");
        }

        TEST_F(MathTest, SoftmaxGrad) {
            float pre[16] = {0, 0, 0, -1.20337,
                             0, 0, -1.07481, 0,
                             0, -476.19, 0, 0,
                             0, -1.02166, 0, 0};
            float a[] = {0.0414, 0.0152, 0.1125, 0.8310,
                         0.0463, 0.0170, 0.9304, 0.0063,
                         0.8650, 0.0021, 0.0158, 0.1171,
                         0.0179, 0.9788, 0.0024, 0.0009};
            float b[16] = {0};
            float target[] = {0.0414, 0.0152, 0.1125, -0.169,
                              0.0463, 0.0170, -0.0696, 0.0063,
                              0.8650, -0.997899, 0.0158, 0.1171,
                              0.0179, -0.0212, 0.0024, 0.0009};
            SoftmaxGrad(4, 4, a, pre, b);
            PrintMat(b, 4, 4, "softmaxgrad_out");
            checkArrayEqual(b, target, 16);
        }


        TEST_F(MathTest, CrossEntropy) {
            float a[] = {0.1, 0.3, 0.1, 0.5,
                         0.4, 0.2, 0.1, 0.3,
                         0.01, 0.7, 0.13, 0.16,
                         0.1, 0.3, 0.4, 0.2};
            float label[] = {3, 0, 1, 2};

            float out = 0;
            CrossEntropy(16, a, 4, label, &out);
            PrintMat(&out, 1, 1, "crossEntropy_out");
            float target = -log(0.5) - log(0.4) - log(0.7) - log(0.4);
            EXPECT_EQ(out, target / 4);
        }

        TEST_F(MathTest, CrossEntropyGrad) {
            float a[] = {0.0414, 0.0152, 0.1125, 0.8310,
                         0.0463, 0.0170, 0.9304, 0.0063,
                         0.8650, 0.0021, 0.0158, 0.1171,
                         0.0179, 0.9788, 0.0024, 0.0009};
            float label[] = {3, 2, 1, 1};

            float out[16] = {0};
            CrossEntropyGrad(16, a, 4, label, out);
            PrintMat(out, 4, 4, "crossEntropygrad_out");
        }

    }
}
