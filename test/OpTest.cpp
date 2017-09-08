//
// Created by Jarlene on 2017/8/4.
//

#include <gtest/gtest.h>
#include "include/Test.h"
#include <matrix/include/op/AddOp.h>
#include <matrix/include/op/FullConnectedOp.h>
#include <matrix/include/op/ConvolutionOp.h>

namespace matrix {

    namespace Test {

        class OpTest : public ::testing::Test {

        };

        TEST_F(OpTest, VariableOp) {
            OpPtr pro = Registry::Global()->GetOp("variable");
            Context context;
            context.mode = RunMode::kCpu;
            context.phase = Phase::TEST;
            context.type = kFloat;
            Shape shape = ShapeN(784, 128);
            float * result = static_cast<float *>(malloc(sizeof(float) * shape.Size()));
            Blob blob(result);
            std::vector<Blob *> inputs;
            std::vector<Shape *> inShape;
            std::map<std::string, Any> params;
            params["isTrain"] = true;
            Operator *op = pro->CreateOperator(context, inputs, &blob, inShape, &shape, params);
            op->AsyncRun();
        }

        TEST_F(OpTest, AddOp) {
            float a[] = {1, 2, 3, 4, 5, 6};
            float b[] = {2, 3, 4, 5, 6, 7};
            float c[] = {3, 5, 7, 9, 11, 13};

            float res[] = {0, 0, 0, 0, 0, 0};

            OpPtr pro = Registry::Global()->GetOp("add");

            Context context;
            context.mode = RunMode::kCpu;
            context.phase = Phase::TEST;
            Shape shape = ShapeN(2, 3);
            std::vector<Blob *> inputs;
            std::vector<Blob *> outputs;
            Blob ablob(a);
            inputs.push_back(&ablob);
            Blob bblob(b);
            inputs.push_back(&bblob);
            Blob resblob(res);

            std::vector<Shape *> inShape;
            inShape.push_back(&shape);
            inShape.push_back(&shape);

            std::map<std::string, Any> params;

            Operator *op = pro->CreateOperator(context, inputs, &resblob, inShape, &shape, params);

            op->AsyncRun();
            int dim = shape.Size();
            checkArrayEqual<float>(c, res, dim);

        }


        TEST_F(OpTest, FullyConnectedOp) {
            float a[] = {1, 2, 3, 4, 5, 6};
            float b[] = {2, 3, 4, 5, 6, 7}; // 28, 34, 64, 79
            float c[] = {3, 5, 1, 4};

            float d[] = {31, 39, 65, 83};

            float res[] = {0, 0, 0, 0, 0, 0};

            OpPtr pro = Registry::Global()->GetOp("fullConnected");

            Context context;
            context.mode = RunMode::kCpu;
            context.phase = Phase::TEST;

            std::vector<Blob *> inputs;
            std::vector<Blob *> outputs;

            Blob ablob(a);
            Blob bblob(b);
            Blob cblob(c);
            inputs.push_back(&ablob);
            inputs.push_back(&bblob);
            inputs.push_back(&cblob);

            Blob resblob(res);

            std::vector<Shape *> inShape;

            Shape out = ShapeN(2, 2);
            Shape in1 = ShapeN(2, 3);
            Shape in2 = ShapeN(3, 2);

            inShape.push_back(&in1);
            inShape.push_back(&in2);
            inShape.push_back(&out);


            std::map<std::string, Any> params;

            Operator *op = pro->CreateOperator(context, inputs, &resblob, inShape, &out, params);

            op->AsyncRun();
            int dim = out.Size();
            checkArrayEqual<float>(d, res, dim);

        }


        TEST_F(OpTest, ConovolutionOp) {
            float a[] = {1, 1, 1, 0, 0,
                         0, 1, 1, 1, 0,
                         0, 0, 1, 1, 1,
                         0, 0, 1, 1, 0,
                         0, 1, 1, 0, 0};

            float kernel[] = {1, 1, 1,
                              1, 1, 0,
                              1, 0, 0};


            float b[] = {1, 2, 3,
                         1, 2, 1,
                         2, 1, 3};

            float c[] = {5, 6, 7,
                         3, 6, 6,
                         3, 5, 9};

            float res[] = {0, 0, 0,
                           0, 0, 0,
                           0, 0, 0};

            OpPtr pro = Registry::Global()->GetOp("convolution");

            Context context;
            context.mode = RunMode::kCpu;
            context.phase = Phase::TEST;

            std::vector<Blob *> inputs;
            std::vector<Blob *> outputs;

            Blob data(a);
            Blob weight(kernel);
            Blob bias(b);

            inputs.push_back(&data);
//            inputs.push_back(&weight);
//            inputs.push_back(&bias);

            Blob outBlob(res);

            std::vector<Shape *> inShape;
            std::vector<Shape *> outShape;

            Shape out = ShapeN(2, 2);
            Shape in1 = ShapeN(1, 1, 5, 5);
            Shape in2 = ShapeN(3, 3);
            inShape.push_back(&in1);
//            inShape.push_back(&in2);
//            inShape.push_back(&in2);



            std::map<std::string, Any> params;
            params["filter_num"] = 1;
            params["filter"] = in2;
            params["bias"] = true;

            Operator *op = pro->CreateOperator(context, inputs, &outBlob, inShape, &out, params);

            op->AsyncRun();
            int dim = out.Size();
//            checkArrayEqual<float>(c, res, dim);
        }


    }
}