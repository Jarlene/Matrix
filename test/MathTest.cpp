//
// Created by Jarlene on 2017/7/24.
//
#include <gtest/gtest.h>
#include "include/Test.h"
#include <matrix/include/base/Tensor.h>
#include <matrix/include/utils/MathTensor.h>
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

    }
}
