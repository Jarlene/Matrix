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
            Context context = Context::Test();
            Shape shape = ShapeN(5, 4);
            float *result = static_cast<float *>(malloc(sizeof(float) * shape.Size()));
            std::vector<void *> inputs;
            std::vector<Shape *> inShape;
            std::map<std::string, Any> params;
            params["isTrain"] = true;
            Operator *op = pro->CreateOperator(context,  &inShape, &shape, params);
            op->SetData(&inputs, result);
            op->AsyncRun();
            PrintMat(result, shape[0], shape[1], "VariableOp_test_result");
            delete result;
        }

        TEST_F(OpTest, AddOp) {
            float a[] = {1, 2, 3, 4, 5, 6};
            float b[] = {2, 3, 4, 5, 6, 7};
            float c[] = {3, 5, 7, 9, 11, 13};

            float res[] = {0, 0, 0, 0, 0, 0};

            OpPtr pro = Registry::Global()->GetOp("add");
            Context context = Context::Test();
            Shape shape = ShapeN(2, 3);
            std::vector<void *> inputs;
            inputs.push_back(a);
            inputs.push_back(b);

            std::vector<Shape *> inShape;
            inShape.push_back(&shape);
            inShape.push_back(&shape);

            Shape out;

            std::map<std::string, Any> params;

            Operator *op = pro->CreateOperator(context, &inShape, &out, params);
            op->SetData(&inputs, res);
            op->AsyncRun();
            int dim = out.Size();
            checkArrayEqual<float>(c, res, dim);
            PrintMat(res, out[0], out[1], "AddOp_test_result");
        }

        TEST_F(OpTest, MulOp) {
            float a[] = {1, 2, 3, 4, 5, 6};
            float b[] = {2, 3, 4, 5, 6, 7}; // 28, 34, 64, 79
            float c[4] = {0};

            float res[4] = {28, 34, 64, 79};
            OpPtr pro = Registry::Global()->GetOp("mul");

            Context context = Context::Test();

            std::vector<void *> inputs;


            inputs.push_back(a);
            inputs.push_back(b);

            std::vector<Shape *> inShape;
            Shape in1 = ShapeN(2, 3);
            Shape in2 = ShapeN(3, 2);
            inShape.push_back(&in1);
            inShape.push_back(&in2);

            std::map<std::string, Any> params;
            Shape out;
            Operator *op = pro->CreateOperator(context, &inShape, &out, params);
            op->SetData(&inputs, c);
            op->AsyncRun();
            PrintMat(c, out[0], out[1], "MulOp_test_result");
            checkArrayEqual<float>(c, res, out.Size());
        }

        TEST_F(OpTest, FullyConnectedOp) {
            float a[] = {1, 2, 3, 4, 5, 6};
            float b[] = {2, 3, 4, 5, 6, 7}; // 28, 34, 64, 79
            float c[] = {3, 5, 1, 4};

            float d[] = {31, 39, 65, 83};

            float res[] = {0, 0, 0, 0, 0, 0};

            OpPtr pro = Registry::Global()->GetOp("fullConnected");

            Context context = Context::Test();

            std::vector<void *> inputs;

            inputs.push_back(a);
            inputs.push_back(b);
            inputs.push_back(c);


            std::vector<Shape *> inShape;

            Shape out = ShapeN(2, 2);
            Shape in1 = ShapeN(2, 3);
            Shape in2 = ShapeN(3, 2);

            inShape.push_back(&in1);
            inShape.push_back(&in2);
            inShape.push_back(&out);


            std::map<std::string, Any> params;

            Operator *op = pro->CreateOperator(context,  &inShape, &out, params);
            op->SetData(&inputs, res);
            op->AsyncRun();
            int dim = out.Size();
            checkArrayEqual<float>(d, res, dim);
            PrintMat(res, out[0], out[1], "FullyConnectedOp_testa_result");
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
                            1};
            float outdata[12] = {0};

            Context context = Context::Test();
            std::vector<void *> inputs;

            inputs.push_back(pre_grad);
            inputs.push_back(outdata);
            inputs.push_back(data);
            inputs.push_back(weight);
            inputs.push_back(bias);


            std::vector<Shape *> inShape;

            Shape pre_grad_shape = ShapeN(4, 3);
            Shape out_shape = ShapeN(4, 3);
            Shape data_shape = ShapeN(4, 2);
            Shape weight_shape = ShapeN(2, 3);
            Shape bias_shape = ShapeN(3);
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
            Operator *op = pro->CreateOperator(context,  &inShape, &out, params);
            if (index == 0) {
                float target[8] = {0};
                float res[8] = {20, 38,
                                11, 26,
                                11, 23,
                                15, 30};

                op->SetData(&inputs, &target);
                op->AsyncRun();
                PrintMat(target, 4, 2, "FullyConnectedGradOp_test_result_data");
                checkArrayEqual(res, target, out.Size());
            } else if (index == 1) {
                float target[6] = {0};
                float res[6] = {37, 17, 22,
                                46, 22, 28};
                op->SetData(&inputs, &target);
                op->AsyncRun();
                PrintMat(target, 2, 3, "FullyConnectedGradOp_test_result_weight");
                checkArrayEqual(res, target, out.Size());
            } else if (index == 2) {
                float target[3] = {0};
                float res[3] = {9, 5, 6};
                op->SetData(&inputs, &target);
                op->AsyncRun();
                PrintMat(target, 3, 1, "FullyConnectedGradOp_test_result_bias");
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


            float b[] = {1};

            float col[81] = {0};

            float c[] = {5, 5, 5,
                         3, 5, 6,
                         2, 5, 7};

            float res[9] = {0};

            OpPtr pro = Registry::Global()->GetOp("convolution");

            Context context = Context::Test();

            std::vector<void *> inputs;



            inputs.push_back(a);
            inputs.push_back(kernel);
            inputs.push_back(b);
            inputs.push_back(col);


            std::vector<Shape *> inShape;
            Shape out;

            Shape dataShape = ShapeN(1, 1, 5, 5);
            Shape filterShape = ShapeN(1, 1, 3, 3);
            Shape biasShape = ShapeN(1);
            Shape cols = ShapeN(1, 9, 9);
            inShape.push_back(&dataShape);
            inShape.push_back(&filterShape);
            inShape.push_back(&biasShape);
            inShape.push_back(&cols);

            std::map<std::string, Any> params;
            Operator *op = pro->CreateOperator(context, &inShape, &out, params);
            op->SetData(&inputs, res);
            op->AsyncRun();
            int dim = out.Size();
            PrintMat(res, 3, 3, "ConovolutionOp_tesst_result");
            checkArrayEqual<float>(c, res, dim);
        }

        TEST_F(OpTest, MulitChannelConovolutionOp) {
            float data[] = {1, 1, 1, 0, 0,
                            0, 1, 1, 1, 0,
                            0, 0, 1, 1, 1,
                            0, 0, 1, 1, 0,
                            0, 1, 1, 0, 0,
                            1, 1, 1, 0, 0,
                            0, 1, 1, 1, 0,
                            0, 0, 1, 1, 1,
                            0, 0, 1, 1, 0,
                            0, 1, 1, 0, 0,
                            1, 1, 1, 0, 0,
                            0, 1, 1, 1, 0,
                            0, 0, 1, 1, 1,
                            0, 0, 1, 1, 0,
                            0, 1, 1, 0, 0};

            float kernel[] = {1, 1, 1,
                              1, 1, 0,
                              1, 0, 0,
                              1, 1, 1,
                              1, 1, 0,
                              1, 0, 0,
                              1, 1, 1,
                              1, 1, 0,
                              1, 0, 0,
                              1, 1, 1,
                              1, 1, 0,
                              1, 0, 0,
                              1, 1, 1,
                              1, 1, 0,
                              1, 0, 0,
                              1, 1, 1,
                              1, 1, 0,
                              1, 0, 0,
                              1, 1, 1,
                              1, 1, 0,
                              1, 0, 0,
                              1, 1, 1,
                              1, 1, 0,
                              1, 0, 0,
                              1, 1, 1,
                              1, 1, 0,
                              1, 0, 0};


            float b[3] = {1, 1, 0};



            float col[81 * 3] = {0};

            float target[] = {13, 13, 12,
                              7, 13, 15,
                              4, 13, 18,
                              13, 13, 12,
                              7, 13, 15,
                              4, 13, 18,
                              13, 13, 12,
                              7, 13, 15,
                              4, 13, 18,};

            float res[9*3] = {0};

            OpPtr pro = Registry::Global()->GetOp("convolution");

            Context context = Context::Test();

            std::vector<void *> inputs;



            inputs.push_back(data);
            inputs.push_back(kernel);
            inputs.push_back(b);
            inputs.push_back(col);


            std::vector<Shape *> inShape;
            Shape out;

            Shape dataShape = ShapeN(1, 3, 5, 5);
            Shape filterShape = ShapeN(3, 3, 3, 3);
            Shape biasShape = ShapeN(3);
            Shape cols = ShapeN(3, 9, 9);
            inShape.push_back(&dataShape);
            inShape.push_back(&filterShape);
            inShape.push_back(&biasShape);
            inShape.push_back(&cols);

            std::map<std::string, Any> params;
            params["filter_num"] = 3;
            Operator *op = pro->CreateOperator(context, &inShape, &out, params);
            op->SetData(&inputs, res);
            op->AsyncRun();
            int dim = out.Size();
            PrintMat(res, 9, 3, "MulitChannelConovolutionOp_test_result");
            checkArrayEqual<float>(target, res, dim);
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
            float col[9*9] = {0};


            OpPtr pro = Registry::Global()->GetOp("grad_convolution");

            Context context = Context::Test();

            std::vector<void *> inputs;

            inputs.push_back(pre_grad);
            inputs.push_back(data);
            inputs.push_back(inputdata);
            inputs.push_back(kernel);
            inputs.push_back(b);
            inputs.push_back(col);

            std::vector<Shape *> inShape;

            Shape inshape = ShapeN(1, 1, 5, 5);
            Shape preg = ShapeN(1, 1, 3, 3);
            Shape self_out = ShapeN(1, 1, 3, 3);
            Shape k = ShapeN(1, 1, 3, 3);
            Shape bias_shape = ShapeN(1);
            Shape cols = ShapeN(1, 9, 9);
            inShape.push_back(&preg);
            inShape.push_back(&self_out);
            inShape.push_back(&inshape);
            inShape.push_back(&k);
            inShape.push_back(&bias_shape);
            inShape.push_back(&cols);
            Shape out;

            std::map<std::string, Any> params;
            int inputIdx = 0;

            params["input_idx"] = inputIdx;
            Operator *op = pro->CreateOperator(context,  &inShape, &out, params);
            if (inputIdx == 0) {

                op->SetData(&inputs, input_grad);
                op->AsyncRun();
                int dim = out.Size();
                float target[25] = {1, 2, 3, 2, 1,
                                    3, 5, 5, 2, 0,
                                    4, 5, 4, 1, 1,
                                    3, 2, 1, 1, 0,
                                    1, 0, 1, 0, 0};
                checkArrayEqual<float>(target, input_grad, dim);
                PrintMat(input_grad, inshape[2], inshape[3], "ConvolutionGradOp_test_result_data");
            } else if (inputIdx == 1) {
                op->SetData(&inputs, filter_grad);
                op->AsyncRun();
                int dim = out.Size();
                float target[9] = {5, 6, 6,
                                   3, 5, 6,
                                   2, 4, 7};
                checkArrayEqual<float>(filter_grad, target, dim);
                PrintMat(filter_grad, out[2], out[3], "ConvolutionGradOp_test_result_kernel");
            } else if (inputIdx == 2) {
                op->SetData(&inputs, bias_grad);
                op->AsyncRun();
                int dim = out.Size();
                float target = 8;
                checkArrayEqual<float>(bias_grad, &target, dim);
                PrintMat(bias_grad, 1, 1, "ConvolutionGradOp_test_result_data_bias");
            }


        }



        TEST_F(OpTest, MultiChannelConvolutionGradOp) {

            float pre_grad[] = {1, 1, 1,
                                2, 1, 0,
                                1, 0, 1,

                                1, 1, 1,
                                2, 1, 0,
                                1, 0, 1,

                                1, 1, 1,
                                2, 1, 0,
                                1, 0, 1,
            };

            float data[] = {
                    4, 4, 4,
                    2, 4, 5,
                    1, 4, 6,

                    4, 4, 4,
                    2, 4, 5,
                    1, 4, 6,

                    4, 4, 4,
                    2, 4, 5,
                    1, 4, 6
            };

            float inputdata[] = {1, 1, 1, 0, 0,
                                 0, 1, 1, 1, 0,
                                 0, 0, 1, 1, 1,
                                 0, 0, 1, 1, 0,
                                 0, 1, 1, 0, 0,

                                 1, 1, 1, 0, 0,
                                 0, 1, 1, 1, 0,
                                 0, 0, 1, 1, 1,
                                 0, 0, 1, 1, 0,
                                 0, 1, 1, 0, 0,

                                 1, 1, 1, 0, 0,
                                 0, 1, 1, 1, 0,
                                 0, 0, 1, 1, 1,
                                 0, 0, 1, 1, 0,
                                 0, 1, 1, 0, 0};

            float kernel[] = {1, 1, 1,
                              1, 1, 0,
                              1, 0, 0,

                              1, 1, 1,
                              1, 1, 0,
                              1, 0, 0,

                              1, 1, 1,
                              1, 1, 0,
                              1, 0, 0,

                              1, 1, 1,
                              1, 1, 0,
                              1, 0, 0,

                              1, 1, 1,
                              1, 1, 0,
                              1, 0, 0,

                              1, 1, 1,
                              1, 1, 0,
                              1, 0, 0,

                              1, 1, 1,
                              1, 1, 0,
                              1, 0, 0,

                              1, 1, 1,
                              1, 1, 0,
                              1, 0, 0,

                              1, 1, 1,
                              1, 1, 0,
                              1, 0, 0};

            float b[] = {1, 1, 0};

            float input_grad[25*3] = {0};

            float filter_grad[27*3] = {0};
            float bias_grad[3] = {0};
            float col[9*9*3] = {0};


            OpPtr pro = Registry::Global()->GetOp("grad_convolution");

            Context context = Context::Test();

            std::vector<void *> inputs;

            inputs.push_back(pre_grad);
            inputs.push_back(data);
            inputs.push_back(inputdata);
            inputs.push_back(kernel);
            inputs.push_back(b);
            inputs.push_back(col);

            std::vector<Shape *> inShape;

            Shape inshape = ShapeN(1, 3, 5, 5);
            Shape preg = ShapeN(1, 3, 3, 3);
            Shape self_out = ShapeN(3, 3, 3, 3);
            Shape k = ShapeN(3, 3, 3, 3);
            Shape biasShape = ShapeN(3);
            Shape cols = ShapeN(3, 9, 9);
            inShape.push_back(&preg);
            inShape.push_back(&self_out);
            inShape.push_back(&inshape);
            inShape.push_back(&k);
            inShape.push_back(&biasShape);
            inShape.push_back(&cols);
            Shape out;

            std::map<std::string, Any> params;
            int inputIdx = 0;
            params["filter_num"] = 3;
            params["input_idx"] = inputIdx;
            Operator *op = pro->CreateOperator(context,  &inShape, &out, params);
            if (inputIdx == 0) {

                op->SetData(&inputs, input_grad);
                op->AsyncRun();
                int dim = out.Size();
                float target[25 * 5] = {3, 6, 9, 6, 3,
                                        9, 15, 15, 6, 0,
                                        12, 15, 12, 3, 3,
                                        9, 6, 3, 3, 0,
                                        3, 0, 3, 0, 0,
                                        3, 6, 9, 6, 3,
                                        9, 15, 15, 6, 0,
                                        12, 15, 12, 3, 3,
                                        9, 6, 3, 3, 0,
                                        3, 0, 3, 0, 0,
                                        3, 6, 9, 6, 3,
                                        9, 15, 15, 6, 0,
                                        12, 15, 12, 3, 3,
                                        9, 6, 3, 3, 0,
                                        3, 0, 3, 0, 0};

                PrintMat(input_grad, inshape[0] * inshape[1] * inshape[2], inshape[3], "MultiChannelConvolutionGradOp_test_result_data");
                checkArrayEqual<float>(target, input_grad, dim);
            } else if (inputIdx == 1) {
                op->SetData(&inputs, filter_grad);
                op->AsyncRun();
                int dim = out.Size();
                float target[27 * 3] = {5, 6, 6,
                                        3, 5, 6,
                                        2, 4, 7,
                                        5, 6, 6,
                                        3, 5, 6,
                                        2, 4, 7,
                                        5, 6, 6,
                                        3, 5, 6,
                                        2, 4, 7,
                                        5, 6, 6,
                                        3, 5, 6,
                                        2, 4, 7,
                                        5, 6, 6,
                                        3, 5, 6,
                                        2, 4, 7,
                                        5, 6, 6,
                                        3, 5, 6,
                                        2, 4, 7,
                                        5, 6, 6,
                                        3, 5, 6,
                                        2, 4, 7,
                                        5, 6, 6,
                                        3, 5, 6,
                                        2, 4, 7,
                                        5, 6, 6,
                                        3, 5, 6,
                                        2, 4, 7,};
                PrintMat(filter_grad, k[0]*k[1]*k[2], k[3], "MultiChannelConvolutionGradOp_test_result_kernel");
                checkArrayEqual<float>(filter_grad, target, dim);
            } else if (inputIdx == 2) {
                op->SetData(&inputs, bias_grad);
                op->AsyncRun();
                int dim = out.Size();
                float target[3] = {12,6,6};
                PrintMat(bias_grad, biasShape[0], 1, "MultiChannelConvolutionGradOp_test_result_bias");
                checkArrayEqual<float>(bias_grad, target, dim);
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
            float maxIndex[9] = {0};

            Context context = Context::Test();

            auto pro = Registry::Global()->GetOp("pooling");
            std::vector<Shape *> inShape;
            auto in = ShapeN(1, 1, 4, 4);
            auto cols = ShapeN(1, 1, 3, 3);
            inShape.push_back(&in);
            inShape.push_back(&cols);
            Shape out;
            std::map<std::string, Any> params;
            params["filter"] = ShapeN(2, 2);
            params["type"] = PoolType::kMax;

            std::vector<void *> inputs;
            inputs.push_back(a);
            inputs.push_back(maxIndex);
            Operator *op = pro->CreateOperator(context,  &inShape, &out, params);
            op->SetData(&inputs, b);
            op->AsyncRun();
            int dim = out.Size();

            PrintMat(b, 3, 3, "PoolingOp_test_result");
            if (get<PoolType>(params["type"]) == kMax) {
                checkArrayEqual<float>(b, c, dim);
                PrintMat(maxIndex, 1, 9, "PoolingOp_max_index");
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

            float index[] = {4, 6, 6, 8, 6, 6, 8, 14, 14};

            std::map<std::string, Any> params;
            params["filter"] = ShapeN(2, 2);
            params["type"] = PoolType::kMax;

            Context context = Context::Test();

            auto pro = Registry::Global()->GetOp("grad_pooling");

            std::vector<void *> inputs;
            inputs.push_back(pre_grad);
            inputs.push_back(out);
            inputs.push_back(input);
            inputs.push_back(index);

            std::vector<Shape *> inShape;
            auto preShape = ShapeN(1, 1, 3, 3);
            auto outShape = ShapeN(1, 1, 3, 3);
            auto inputShape = ShapeN(1, 1, 4, 4);
            inShape.push_back(&preShape);
            inShape.push_back(&outShape);
            inShape.push_back(&inputShape);
            inShape.push_back(&outShape);

            Shape out_Shape;

            Operator *op = pro->CreateOperator(context,  &inShape, &out_Shape, params);
            op->SetData(&inputs, b);
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
            PrintMat(b, 4, 4, "PoolingGradOp_test_result");


        }

        TEST_F(OpTest, ActivationOp) {
            float a[] = {1, 2, 3, 4, 5, 6};
            float b[6] = {0};
            Shape in_shape = ShapeN(3,2);

            Context context = Context::Test();

            std::vector<void *> inputs;
            inputs.push_back(a);
            std::vector<Shape *> inShape;
            inShape.push_back(&in_shape);
            OpPtr pro = Registry::Global()->GetOp("activation");

            std::map<std::string, Any> params;
            ActType type = kRelu;
            params["type"] = type;
            Shape out;
            Operator *op = pro->CreateOperator(context,  &inShape, &out, params);
            op->SetData(&inputs, b);
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
            PrintMat(b, out[0], out[1], "ActivationOp_test_result");
        }

        TEST_F(OpTest, ActivationGradOp) {
            float pre_grad[] = {1, 0.2, 0.4, 2, 3, 1};
            float a[] = {1, 2, 3, 4, 5, 6};
            float b[6] = {0.731059, 0.880797,
                          0.952574, 0.982014,
                          0.993307, 0.997527};

            float c[6] = {0};


            Context context = Context::Test();

            std::vector<void *> inputs;
            inputs.push_back(a);
            inputs.push_back(pre_grad);
            inputs.push_back(b);

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
            Operator *op = pro->CreateOperator(context, &inShape, &out, params);
            op->SetData(&inputs, c);
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
            PrintMat(c, out[0], out[1], "ActivationGradOp_test_result");
        }

        TEST_F(OpTest, LossOp) {
            float input[] = {0.2, 0.3,
                             0.1, 0.4,
                             0.1, 0.5};
            float label[] = {1 ,0, 1};
            Shape input_shape = ShapeN(3, 2);
            Shape label_shape = ShapeN(3);

            float a[] = {0};


            Context context = Context::Test();

            std::vector<void *> inputs;
            inputs.push_back(input);
            inputs.push_back(label);



            std::vector<Shape *> inShape;
            inShape.push_back(&input_shape);
            inShape.push_back(&label_shape);

            OpPtr pro = Registry::Global()->GetOp("loss");

            std::map<std::string, Any> params;
            auto type = kCrossEntropy;
            params["type"] = type;
            Shape out;
            Operator *op = pro->CreateOperator(context,  &inShape, &out, params);
            op->SetData(&inputs, a);
            op->AsyncRun();

            PrintMat(a, 1, 1, "LossOp_test_result");
            float target = (-log(0.3f) - log(0.1f) -log(0.5f)) / 3;
            EXPECT_FLOAT_EQ(a[0], target);
        }
    }
}