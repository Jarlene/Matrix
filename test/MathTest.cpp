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


        TEST_F(MathTest, Img2Col) {
            float a[] = {1, 1, 1, 0, 0,
                         0, 1, 1, 1, 0,
                         0, 0, 1, 1, 1,
                         0, 0, 1, 1, 0,
                         0, 1, 1, 0, 0};

            float kernel[] = {1, 1, 1,
                              1, 1, 0,
                              1, 0, 0};

            float colBuffer[81] = {0};

            float b[9];

            float c[] = {4, 4, 4,
                         2, 4, 5,
                         1, 4, 6};

            Shape input = ShapeN(1, 1, 5, 5);
            Shape kShape = ShapeN(1, 3, 3);
            Shape stride = ShapeN(1, 1);
            Shape padding = ShapeN(0, 0);
            Shape dilation = ShapeN(1, 1);
            Shape out = ShapeN(1, 1, 3, 3);

            const int input_offset = 25;
            const int output_offset = 9;
            int M = 1;
            int N = 9;
            int K = 9;
            const int filter_offset = 9;
            for (int i = 0; i < 1; ++i) {
                for (int j = 0; j < 1; ++j) {
                    img2col(a, 1, 5, 5, 1, 1, 0, 0, 3, 3, 1, 1, colBuffer);
                    CPUGemm<float>(NoTrans, NoTrans, M, N, K, 1.0f, kernel + j * filter_offset, colBuffer,
                                   0.0f, b + j * output_offset);
                }
            }

            checkArrayEqual<float>(b, c, 9);
        }

        TEST_F(MathTest, Col2Img) {
            float inputData[] = {1, 1, 1,
                                 2, 1, 0,
                                 1, 0, 1};

            float kernelData[] = {1, 1, 1,
                                  1, 1, 0,
                                  1, 0, 0};

            float colBuffer[45] = {0};

            float a[25] = {0};

            float res[25] = {0,0,1,1,1,
                             0,1,4,3,1,
                             1,4,7,3,2,
                             2,4,4,2,1,
                             1,1,2,1,1};

            Shape input = ShapeN(1, 1, 3, 3);
            Shape kShape = ShapeN(1, 3, 3);
            Shape stride = ShapeN(1, 1);
            Shape padding = ShapeN(2, 2);
            Shape dilation = ShapeN(1, 1);
            Shape out = ShapeN(1, 1, 5, 5);

            int M = 1 / 1 * 3 * 3;
            int N = 3 * 3;
            int K = 1 / 1;

            CPUGemm<float>(Trans, NoTrans, M, N, K, 1.0f,
                           kernelData,
                           inputData,
                           0.0f, colBuffer);

            col2img(a, 1, 5, 5, 1, 1, 2, 2, 3, 3, 1, 1, colBuffer);
            for (int i = 0; i < 25; ++i) {
                std::cout << a[i] << std::endl;
            }
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
    }
}
