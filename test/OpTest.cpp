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

            OpPtr pro =  Registry::Global()->GetOp("add");

            Context context;
            context.mode = RunMode::kCpu;
            context.phase = Phase::TEST;
            Shape shape = ShapeN(2, 3);
            std::vector<Blob> inputs;
            std::vector<Blob> outputs;
            inputs.push_back(Blob(a));
            inputs.push_back(Blob(b));
            outputs.push_back(Blob(res));

            std::vector<Shape> inShape;
            inShape.push_back(shape);
            inShape.push_back(shape);
            std::vector<Shape> outShape;
            outShape.push_back(shape);

            std::map<std::string, Any> params;

            Operator* op = pro->CreateOperator(context, inputs, outputs, inShape, outShape, params);

            op->AsyncRun();
            int dim = shape.Size();
            checkArrayEqual<float>(c, res, dim);

        }


    }
}