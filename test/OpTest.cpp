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
            float *result = static_cast<float *>(malloc(sizeof(float) * shape.Size()));
            Blob blob(result);
            std::vector<Blob *> inputs;
            std::vector<Shape *> inShape;
            std::map<std::string, Any> params;
            params["isTrain"] = true;
            Operator *op = pro->CreateOperator(context, inputs, &blob, inShape, &shape, params);
            op->AsyncRun();
            delete result;
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

        TEST_F(OpTest, MulOp) {
            float a[] = {1, 2, 3, 4, 5, 6};
            float b[] = {2, 3, 4, 5, 6, 7}; // 28, 34, 64, 79
            float c[4] = {0};

            float res[4] = {28, 34, 64, 79};
            OpPtr pro = Registry::Global()->GetOp("mul");

            Context context;
            context.mode = RunMode::kCpu;
            context.phase = Phase::TEST;

            std::vector<Blob *> inputs;
            Blob ablob(a);
            Blob bblob(b);

            inputs.push_back(&ablob);
            inputs.push_back(&bblob);

            std::vector<Shape *> inShape;
            Shape in1 = ShapeN(2, 3);
            Shape in2 = ShapeN(3, 2);
            inShape.push_back(&in1);
            inShape.push_back(&in2);

            std::map<std::string, Any> params;
            Shape out;
            Blob cblob(c);
            Operator *op = pro->CreateOperator(context, inputs, &cblob, inShape, &out, params);

            op->AsyncRun();
            PrintMat(c, out[0], out[1], "mul");
            checkArrayEqual<float>(c, res, out.Size());
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

        TEST_F(OpTest, FullyConnectedGradOp) {
            float pre_grad[] = {1, 2, 3,
                                4, 1, 0,
                                2, 1, 1,
                                2, 1, 2};

            float data[] = {1, 2,
                            3, 4,
                            5, 6,
                            7, 8};
            float weight[] = {2, 3, 4,
                              5, 6, 7}; // 28, 34, 64, 79
            float bias[] = {3,
                            5,
                            1,
                            4};
            float outdata[12] = {0};

            Context context;
            context.mode = RunMode::kCpu;
            context.phase = Phase::TEST;
            std::vector<Blob *> inputs;
            Blob pre_grad_blob(pre_grad);
            Blob out_blob(outdata);
            Blob data_blob(data);
            Blob weight_blob(weight);
            Blob bias_blob(bias);

            inputs.push_back(&pre_grad_blob);
            inputs.push_back(&out_blob);
            inputs.push_back(&data_blob);
            inputs.push_back(&weight_blob);
            inputs.push_back(&bias_blob);


            std::vector<Shape *> inShape;

            Shape pre_grad_shape = ShapeN(4, 3);
            Shape out_shape = ShapeN(4, 3);
            Shape data_shape = ShapeN(4, 2);
            Shape weight_shape = ShapeN(2, 3);
            Shape bias_shape = ShapeN(4);
            inShape.push_back(&pre_grad_shape);
            inShape.push_back(&out_shape);
            inShape.push_back(&data_shape);
            inShape.push_back(&weight_shape);
            inShape.push_back(&bias_shape);

            OpPtr pro = Registry::Global()->GetOp("grad_fullConnected");
            Shape out;
            std::map<std::string, Any> params;
            int index = 2;
            params["input_idx"] = index;
            if (index == 0) {
                float target[8] = {0};
                float res[8] = {20, 38,
                                11, 26,
                                11, 23,
                                15, 30};
                Blob resblob(target);
                Operator *op = pro->CreateOperator(context, inputs, &resblob, inShape, &out, params);
                op->AsyncRun();
                PrintMat(target, 4, 2, "data_grad");
                checkArrayEqual(res, target, out.Size());
            } else if (index == 1) {
                float target[6] = {0};
                float res[6] = {37, 17, 22,
                                46, 22, 28};
                Blob resblob(target);
                Operator *op = pro->CreateOperator(context, inputs, &resblob, inShape, &out, params);
                op->AsyncRun();
                PrintMat(target, 2, 3, "weight_grad");
                checkArrayEqual(res, target, out.Size());
            } else if (index == 2) {
                float target[4] = {0};
                float res[4] = {6, 5, 4, 5};
                Blob resblob(target);
                Operator *op = pro->CreateOperator(context, inputs, &resblob, inShape, &out, params);
                op->AsyncRun();
                PrintMat(target, 4, 1, "bias_grad");
                checkArrayEqual(res, target, out.Size());
            }


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


            Blob data(a);
            Blob weight(kernel);
            Blob bias(b);

            inputs.push_back(&data);
            inputs.push_back(&weight);
            inputs.push_back(&bias);

            Blob outBlob(res);

            std::vector<Shape *> inShape;
            Shape out = ShapeN(2, 2);
            Shape in1 = ShapeN(1, 1, 5, 5);
            Shape in2 = ShapeN(3, 3);
            inShape.push_back(&in1);
            inShape.push_back(&in2);
            inShape.push_back(&in2);


            std::map<std::string, Any> params;
            params["filter_num"] = 1;
//            params["filter"] = in2;
//            params["bias"] = true;

            Operator *op = pro->CreateOperator(context, inputs, &outBlob, inShape, &out, params);

            op->AsyncRun();
            int dim = out.Size();
            checkArrayEqual<float>(c, res, dim);
        }

        TEST_F(OpTest, ConvolutionGradOp) {

            float pre_grad[] = {1, 1, 1,
                                2, 1, 0,
                                1, 0, 1};

            float data[] = {
                    4, 4, 4,
                    2, 4, 5,
                    1, 4, 6
            };

            float inputdata[] = {1, 1, 1, 0, 0,
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

            float input_grad[25] = {0};

            float filter_grad[9] = {0};
            float bias_grad[9] = {0};


            OpPtr pro = Registry::Global()->GetOp("grad_convolution");

            Context context;
            context.mode = RunMode::kCpu;
            context.phase = Phase::TEST;

            std::vector<Blob *> inputs;


            Blob preGrad(pre_grad);
            Blob selfOut(data);
            Blob inputData(inputdata);
            Blob weight(kernel);
            Blob bias(b);


            inputs.push_back(&preGrad);
            inputs.push_back(&selfOut);
            inputs.push_back(&inputData);
            inputs.push_back(&weight);
            inputs.push_back(&bias);

            std::vector<Shape *> inShape;

            Shape inshape = ShapeN(1, 1, 5, 5);
            Shape preg = ShapeN(1, 1, 3, 3);
            Shape self_out = ShapeN(1, 1, 3, 3);
            Shape k = ShapeN(3, 3);
            inShape.push_back(&preg);
            inShape.push_back(&self_out);
            inShape.push_back(&inshape);
            inShape.push_back(&k);
            inShape.push_back(&self_out);

            Shape out;

            std::map<std::string, Any> params;
            params["filter_num"] = 1;
            int inputIdx = 2;

            params["input_idx"] = inputIdx;
            if (inputIdx == 0) {
                Blob outBlob(input_grad);
                Operator *op = pro->CreateOperator(context, inputs, &outBlob, inShape, &out, params);
                op->AsyncRun();
                int dim = out.Size();
                float target[25] = {1, 2, 3, 2, 1,
                                    3, 5, 5, 2, 0,
                                    4, 5, 4, 1, 1,
                                    3, 2, 1, 1, 0,
                                    1, 0, 1, 0, 0};
                checkArrayEqual<float>(target, input_grad, dim);
                PrintMat(input_grad, inshape[2], inshape[3], "input_grad");
            } else if (inputIdx == 1) {
                Blob outBlob(filter_grad);
                Operator *op = pro->CreateOperator(context, inputs, &outBlob, inShape, &out, params);
                op->AsyncRun();
                int dim = out.Size();
                float target[9] = {5, 6, 6,
                                   3, 5, 6,
                                   2, 4, 7};
                checkArrayEqual<float>(filter_grad, target, dim);
                PrintMat(filter_grad, k[0], k[1], "input_filter");
            } else if (inputIdx == 2) {
                Blob outBlob(bias_grad);
                Operator *op = pro->CreateOperator(context, inputs, &outBlob, inShape, &out, params);
                op->AsyncRun();
                int dim = out.Size();
                checkArrayEqual<float>(bias_grad, pre_grad, dim);
                PrintMat(bias_grad, preg[2], preg[3], "input_bias");
            }


        }

        TEST_F(OpTest, PoolingOp) {
            float a[] = {1, 3, 1, 5,
                         4, 2, 6, 2,
                         7, 1, 3, 5,
                         1, 3, 10, 2};

            float b[9] = {0};

            float c[9] = {4, 6, 6,
                          7, 6, 6,
                          7, 10, 10};

            Context context;
            context.mode = RunMode::kCpu;
            context.phase = Phase::TEST;

            auto pro = Registry::Global()->GetOp("pooling");
            std::vector<Shape *> inShape;
            auto in = ShapeN(1, 1, 4, 4);
            inShape.push_back(&in);

            Shape out;
            std::map<std::string, Any> params;
            params["filter"] = ShapeN(2, 2);
            params["type"] = PoolType::kAvg;

            Blob blobIn(a);
            Blob blobOut(b);

            std::vector<Blob *> inputs;
            inputs.push_back(&blobIn);
            Operator *op = pro->CreateOperator(context, inputs, &blobOut, inShape, &out, params);
            op->AsyncRun();
            int dim = out.Size();


            PrintMat(b, 3, 3, "pool_out");
            if (get<PoolType>(params["type"]) == kMax) {
                checkArrayEqual<float>(b, c, dim);
                ASSERT_TRUE(params.count("max_index") > 0);
                auto maxIndex = get<Tensor<int>>(params["max_index"]);
                PrintMat(maxIndex.Data(), 1, 9, "max_index");
            } else {
                float target[] = {2.5, 2, 1.75,
                                  2.333333, 1.333333, 1.333333,
                                  1.5, 1.4166666, 1.25};
                checkArrayEqual<float>(b, target, dim);
            }
        }

        TEST_F(OpTest, PoolingGradOp) {

            float b[16] = {0};

            float pre_grad[] = {4, 6, 6,
                                7, 6, 6,
                                7, 10, 10};

            float out[] = {4, 6, 6,
                           7, 6, 6,
                           7, 10, 10};

            float input[] = {1, 3, 1, 5,
                             4, 2, 6, 2,
                             7, 1, 3, 5,
                             1, 3, 10, 2};


            int index[] = {4, 6, 6, 8, 6, 6, 8, 14, 14};

            Tensor<int> maxIndex(index, ShapeN(9));

            std::map<std::string, Any> params;
            params["filter"] = ShapeN(2, 2);
            params["type"] = PoolType::kAvg;
            params["max_index"] = maxIndex;

            Context context;
            context.mode = RunMode::kCpu;
            context.phase = Phase::TEST;

            auto pro = Registry::Global()->GetOp("grad_pooling");

            std::vector<Blob *> inputs;
            Blob pre(pre_grad);
            Blob outB(out);
            Blob inB(input);
            inputs.push_back(&pre);
            inputs.push_back(&outB);
            inputs.push_back(&inB);

            Blob blobOut(b);

            std::vector<Shape *> inShape;
            auto preShape = ShapeN(1, 1, 3, 3);
            auto outShape = ShapeN(1, 1, 3, 3);
            auto inputShape = ShapeN(1, 1, 4, 4);
            inShape.push_back(&preShape);
            inShape.push_back(&outShape);
            inShape.push_back(&inputShape);

            Shape out_Shape;

            Operator *op = pro->CreateOperator(context, inputs, &blobOut, inShape, &out_Shape, params);
            op->AsyncRun();
            int dim = out_Shape.Size();
            if (get<PoolType>(params["type"]) == kMax) {
                float target[16] = {0, 0, 0, 0,
                                    4, 0, 24, 0,
                                    14, 0, 0, 0,
                                    0, 0, 20, 0};
                checkArrayEqual<float>(b, target, dim);
            } else if (get<PoolType>(params["type"]) == kAvg) {
                float target[16] = {1, 2.5, 3, 5,
                                    2.5, 5.5, 6.25, 11.75,
                                    4, 5.5, 3.25, 6.75,
                                    3, 4.25, 2.75, 6.5};
                checkArrayEqual<float>(b, target, dim);
            }
            PrintMat(b, 4, 4, "grad_pool_out");


        }

        TEST_F(OpTest, ActivationOp) {
            float a[] = {1, 2, 3, 4, 5, 6};
            float b[6] = {0};
            Blob in(a);
            Shape in_shape = ShapeN(3,2);

            Context context;
            context.mode = RunMode::kCpu;
            context.phase = Phase::TEST;

            std::vector<Blob *> inputs;
            inputs.push_back(&in);
            std::vector<Shape *> inShape;
            inShape.push_back(&in_shape);
            OpPtr pro = Registry::Global()->GetOp("activation");

            std::map<std::string, Any> params;
            ActType type = kRelu;
            params["type"] = type;
            Shape out;
            Blob outblob(b);
            Operator *op = pro->CreateOperator(context, inputs, &outblob, inShape, &out, params);
            op->AsyncRun();

            switch (type) {
                case kRelu:
                {
                    float target[] = {1, 2, 3, 4, 5, 6};
                    checkArrayEqual(b, target, out.Size());
                }
                    break;

                case kSigmoid:
                {
                    float target[] = {0.731059, 0.880797,
                                      0.952574, 0.982014,
                                      0.993307, 0.997527};
                    checkArrayEqual(b, target, out.Size());
                }
                    break;
                case kTanh:
                {
                    float target[] = {0.761594, 0.964028,
                                      0.995055, 0.999329,
                                      0.999909, 0.999988};
                    checkArrayEqual(b, target, out.Size());
                }
                    break;

            }
            PrintMat(b, out[0], out[1], "activation");
        }

        TEST_F(OpTest, ActivationGradOp) {
            float pre_grad[] = {1, 0.2, 0.4, 2, 3, 1};
            float a[] = {1, 2, 3, 4, 5, 6};
            float b[6] = {0.731059, 0.880797,
                          0.952574, 0.982014,
                          0.993307, 0.997527};

            float c[6] = {0};


            Context context;
            context.mode = RunMode::kCpu;
            context.phase = Phase::TEST;

            Blob in(a);
            Blob pre(pre_grad);
            Blob outBlob(b);
            std::vector<Blob *> inputs;
            inputs.push_back(&pre);
            inputs.push_back(&outBlob);
            inputs.push_back(&in);

            Shape pre_grad_shape = ShapeN(3, 2);
            Shape out_shape = ShapeN(3, 2);
            Shape in_shape = ShapeN(3,2);
            std::vector<Shape *> inShape;
            inShape.push_back(&pre_grad_shape);
            inShape.push_back(&out_shape);
            inShape.push_back(&in_shape);


            OpPtr pro = Registry::Global()->GetOp("grad_activation");

            std::map<std::string, Any> params;
            ActType type = kRelu;
            params["type"] = type;
            Shape out;
            Blob outblob(c);
            Operator *op = pro->CreateOperator(context, inputs, &outblob, inShape, &out, params);
            op->AsyncRun();

            switch (type) {
                case kRelu:
                {
                    float target[] = {1, 1,
                                      1, 1,
                                      1, 1};
                    checkArrayEqual(c, target, out.Size());
                }
                    break;

                case kSigmoid:
                {
                    float target[] = {0.196612, 0.104994,
                                      0.0451768, 0.0176625,
                                      0.00664821, 0.00246688};
                    checkArrayEqual(c, target, out.Size());
                }
                    break;
                case kTanh:
                {
                    float target[] = {0.465553, 0.224197,
                                      0.0926027, 0.0356485,
                                      0.0133412, 0.00493985};
                    checkArrayEqual(c, target, out.Size());
                }
                    break;

            }
            PrintMat(c, out[0], out[1], "grad_activation");
        }
    }
}