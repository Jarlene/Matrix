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
            float  a[] = {0, 1, 2,
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
            pooling2D(a, 1, 1, 4, 4, 1, 1, 0, 0, 2, 2, 1, 1, b, kMax, c);
            PrintMat(b, 1, 9, "out");
            PrintMat(c, 1, 9, "index");

            float target[] = {4, 6, 6, 7, 6, 6, 7, 10, 10};
            float indexTarget[] = {4, 6, 6, 8, 6, 6, 8, 14, 14};
            checkArrayEqual<float>(b, target, 9);
            checkArrayEqual<float>(c, indexTarget, 9);
        }


        TEST_F(MathTest, CrossEntropy) {
            float a[] = {0.1, 0.3, 0.1, 0.5,
                         0.4, 0.2, 0.1, 0.3,
                         0.01, 0.7, 0.13, 0.16,
                         0.1, 0.3, 0.4, 0.2};
            float label[] = {3, 0, 1, 2};

            float out = 0;
            CrossEntropy(16, a, 4, label, &out);
            PrintMat(&out, 1, 1, "out");
            float target = -log(0.5) - log(0.4) - log(0.7) - log(0.4);
            EXPECT_EQ(out, target/4);
        }


    }
}
