//
// Created by 郑珊 on 2017/8/4.
//

#include <gtest/gtest.h>
#include "include/Test.h"
#include <matrix/include/op/AddOp.h>
namespace matrix {

    namespace Test {

        class OpTest : public ::testing::Test {

        };

        TEST_F(OpTest, AddOp) {
            float a[] = {1, 2, 3, 4, 5, 6};
            float b[] = {2, 3, 4, 5, 6, 7};
            float c[] = {3, 5, 7, 9, 11, 13};

            float res[] = {0, 0, 0, 0, 0, 0};

            AddParam param(MatrixType::kFloat);
            param.inShape = ShapeN(2, 3);

            param.in = {Blob(a), Blob(b)};
            Blob blob(res);
            param.out = &blob;

            auto op = CreateOp<CPU>(param);



            op->AsyncRun();
            int dim = param.inShape.Size();
            checkArrayEqual<float>(c, res, dim);

        }


    }
}