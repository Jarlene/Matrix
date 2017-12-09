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


        TEST_F(OpTest, MultiChannelPoolingOp) {
            float a[] = {1, 3, 1, 5,
                         4, 2, 6, 2,
                         7, 1, 3, 5,
                         1, 3, 10, 2,

                         1, 3, 1, 5,
                         4, 2, 6, 2,
                         7, 1, 3, 5,
                         1, 3, 10, 2,

                         1, 3, 1, 5,
                         4, 2, 6, 2,
                         7, 1, 3, 5,
                         1, 3, 10, 2};

            float b[9 * 3] = {0};


            float maxIndex[9 * 3] = {0};

            Context context = Context::Test();

            auto pro = Registry::Global()->GetOp("pooling");
            std::vector<Shape *> inShape;
            auto in = ShapeN(1, 3, 4, 4);
            auto cols = ShapeN(1, 3, 3, 3);
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

            PrintMat(b, 9, 3, "multi_PoolingOp_test_result");
            if (get<PoolType>(params["type"]) == kMax) {
                float c[9 * 3] = {4, 6, 6,
                                  7, 6, 6,
                                  7, 10, 10,

                                  4, 6, 6,
                                  7, 6, 6,
                                  7, 10, 10,

                                  4, 6, 6,
                                  7, 6, 6,
                                  7, 10, 10};
                checkArrayEqual<float>(b, c, dim);
                PrintMat(maxIndex, 9, 3, "PoolingOp_max_index");
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

        TEST_F(OpTest, MutliPoolingGradOp) {

            float b[16*3] = {0};

            float pre_grad[] = {4, 6, 6,
                                7, 6, 6,
                                7, 10, 10,

                                4, 6, 6,
                                7, 6, 6,
                                7, 10, 10,

                                4, 6, 6,
                                7, 6, 6,
                                7, 10, 10};

            float out[] = {4, 6, 6,
                           7, 6, 6,
                           7, 10, 10,

                           4, 6, 6,
                           7, 6, 6,
                           7, 10, 10,

                           4, 6, 6,
                           7, 6, 6,
                           7, 10, 10};

            float input[] = {1, 3, 1, 5,
                             4, 2, 6, 2,
                             7, 1, 3, 5,
                             1, 3, 10, 2,

                             1, 3, 1, 5,
                             4, 2, 6, 2,
                             7, 1, 3, 5,
                             1, 3, 10, 2,

                             1, 3, 1, 5,
                             4, 2, 6, 2,
                             7, 1, 3, 5,
                             1, 3, 10, 2};

            float index[] = {4, 6, 6, 8, 6, 6, 8, 14, 14,
                             4, 6, 6, 8, 6, 6, 8, 14, 14,
                             4, 6, 6, 8, 6, 6, 8, 14, 14};

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
            auto preShape = ShapeN(1, 3, 3, 3);
            auto outShape = ShapeN(1, 3, 3, 3);
            auto inputShape = ShapeN(1, 3, 4, 4);
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
                float target[16 * 3] = {0, 0, 0, 0,
                                    4, 0, 24, 0,
                                    14, 0, 0, 0,
                                    0, 0, 20, 0,

                                    0, 0, 0, 0,
                                    4, 0, 24, 0,
                                    14, 0, 0, 0,
                                    0, 0, 20, 0,

                                    0, 0, 0, 0,
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
            PrintMat(b, 4 * 3, 4, "PoolingGradOp_test_result");


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



        TEST_F(OpTest, AccuracyOp) {

            float data[] = {0.000195468000000000, 3.62903000000000e-05, 0.000805115000000000, 0.0163546000000000,
                            1.88605000000000e-05, 0.00386984000000000, 2.00921000000000e-05, 5.82317000000000e-05,
                            0.796994000000000, 0.181648000000000,
                            6.01613000000000e-06, 3.43495000000000e-06, 5.75160000000000e-05, 4.32065000000000e-05,
                            5.30456000000000e-06, 1.90706000000000e-05, 1.25619000000000e-07, 0.997551000000000,
                            0.000458775000000000, 0.00185570000000000,
                            1.38301000000000e-06, 0.988962000000000, 0.000703259000000000, 0.00641904000000000,
                            4.85611000000000e-06, 0.000387364000000000, 0.000108900000000000, 0.000414172000000000,
                            0.00141117000000000, 0.00158819000000000,
                            3.10793000000000e-05, 0.970408000000000, 0.0142346000000000, 0.00333922000000000,
                            1.97081000000000e-06, 0.000190199000000000, 0.00117058000000000, 5.36525000000000e-05,
                            0.0103309000000000, 0.000239691000000000,
                            0.000672066000000000, 9.94432000000000e-05, 2.96487000000000e-05, 0.948908000000000,
                            4.57272000000000e-06, 0.00175995000000000, 1.12699000000000e-06, 0.00665107000000000,
                            0.00427358000000000, 0.0376003000000000,
                            3.76170000000000e-06, 9.43796000000000e-09, 5.44303000000000e-08, 9.58982000000000e-06,
                            4.91137000000000e-07, 2.19530000000000e-06, 7.27557000000000e-09, 0.999619000000000,
                            1.16953000000000e-05, 0.000352742000000000,
                            2.98151000000000e-05, 0.978785000000000, 0.00184384000000000, 0.00612821000000000,
                            2.33034000000000e-05, 0.00371122000000000, 0.000666945000000000, 0.000904498000000000,
                            0.00608804000000000, 0.00181913000000000,
                            9.23972000000000e-05, 0.948529000000000, 0.0278803000000000, 0.00602487000000000,
                            2.09917000000000e-05, 0.000313456000000000, 0.00226132000000000, 0.000199201000000000,
                            0.0138663000000000, 0.000812391000000000,
                            6.69668000000000e-05, 5.81244000000000e-08, 6.54577000000000e-05, 0.000105597000000000,
                            2.62841000000000e-07, 8.84981000000000e-06, 1.64453000000000e-07, 0.998887000000000,
                            1.44253000000000e-05, 0.000851070000000000,
                            0.000125664000000000, 1.43168000000000e-06, 0.000595432000000000, 2.56333000000000e-06,
                            5.43885000000000e-06, 7.00372000000000e-05, 0.999157000000000, 6.08696000000000e-07,
                            2.55207000000000e-05, 1.61533000000000e-05,
                            5.74075000000000e-06, 0.983344000000000, 0.000607147000000000, 0.00707628000000000,
                            1.31716000000000e-05, 0.00135590000000000, 0.000210725000000000, 0.000506012000000000,
                            0.00341130000000000, 0.00346940000000000,
                            1.21803000000000e-07, 1.08080000000000e-06, 1.70946000000000e-05, 1.05255000000000e-05,
                            1.48282000000000e-07, 2.63931000000000e-07, 2.37509000000000e-09, 0.999805000000000,
                            2.48116000000000e-05, 0.000140759000000000,
                            0.0254782000000000, 2.15806000000000e-05, 0.624529000000000, 0.271850000000000,
                            0.00781360000000000, 0.0471999000000000, 0.0211425000000000, 3.90323000000000e-05,
                            0.000787158000000000, 0.00113943000000000,
                            0.00165053000000000, 0.000452204000000000, 0.00825645000000000, 0.808999000000000,
                            0.00485915000000000, 0.167099000000000, 0.000148602000000000, 6.73833000000000e-05,
                            0.00596729000000000, 0.00250097000000000,
                            0.978597000000000, 4.69575000000000e-05, 0.0141915000000000, 0.00177721000000000,
                            3.70365000000000e-06, 0.000609335000000000, 0.000198882000000000, 0.000321979000000000,
                            0.00409730000000000, 0.000155754000000000,
                            0.000305044000000000, 2.22475000000000e-05, 0.000461942000000000, 0.992796000000000,
                            5.25809000000000e-05, 0.00591072000000000, 3.22938000000000e-06, 9.11856000000000e-06,
                            0.000406859000000000, 3.21066000000000e-05,
                            0.000312508000000000, 6.09679000000000e-05, 0.000165950000000000, 0.993091000000000,
                            8.41449000000000e-07, 0.00621940000000000, 2.77728000000000e-07, 4.22131000000000e-06,
                            0.000143324000000000, 1.48059000000000e-06,
                            0.000141104000000000, 6.38573000000000e-06, 0.000766223000000000, 6.43484000000000e-06,
                            1.54399000000000e-06, 9.09050000000000e-05, 0.998920000000000, 1.44124000000000e-07,
                            5.68220000000000e-05, 1.02384000000000e-05,
                            0.000203198000000000, 0.000128082000000000, 0.00176044000000000, 0.000194214000000000,
                            2.32274000000000e-07, 0.00327485000000000, 4.15088000000000e-06, 2.18300000000000e-06,
                            0.994325000000000, 0.000107486000000000,
                            0.00445589000000000, 0.0409197000000000, 0.0823506000000000, 0.0520841000000000,
                            0.00504618000000000, 0.663361000000000, 0.00235255000000000, 0.00384513000000000,
                            0.138584000000000, 0.00700075000000000,
                            2.90026000000000e-05, 2.14760000000000e-05, 0.000408175000000000, 0.000233742000000000,
                            0.971094000000000, 0.000723132000000000, 0.000320334000000000, 0.00503595000000000,
                            0.000980364000000000, 0.0211537000000000,
                            0.00123501000000000, 4.52749000000000e-06, 0.00290349000000000, 1.43277000000000e-05,
                            0.000321463000000000, 0.00203973000000000, 0.993386000000000, 2.30239000000000e-05,
                            2.92793000000000e-05, 4.34599000000000e-05,
                            0.999983000000000, 4.91634000000000e-11, 3.62515000000000e-06, 2.96862000000000e-08,
                            2.44751000000000e-11, 3.08181000000000e-06, 5.38075000000000e-06, 7.09797000000000e-07,
                            3.92437000000000e-06, 4.70538000000000e-08,
                            7.15968000000000e-07, 8.20111000000000e-08, 0.999938000000000, 3.03092000000000e-05,
                            7.49284000000000e-06, 2.37672000000000e-08, 1.70482000000000e-05, 1.44330000000000e-08,
                            3.40194000000000e-06, 2.72138000000000e-06,
                            0.0138911000000000, 0.123396000000000, 0.192046000000000, 0.00874621000000000,
                            0.0273463000000000, 0.106503000000000, 0.186026000000000, 0.0451840000000000,
                            0.243037000000000, 0.0538250000000000,
                            7.88863000000000e-05, 2.20367000000000e-06, 0.000163029000000000, 1.75248000000000e-05,
                            0.000177070000000000, 0.0206919000000000, 6.62238000000000e-05, 7.04597000000000e-06,
                            0.978659000000000, 0.000137428000000000,
                            0.000117470000000000, 0.00245581000000000, 0.00943069000000000, 0.0368397000000000,
                            0.00639376000000000, 0.0783318000000000, 0.830269000000000, 6.89591000000000e-05,
                            0.00684456000000000, 0.0292480000000000,
                            4.99110000000000e-05, 0.00333913000000000, 0.00182268000000000, 0.000826841000000000,
                            3.14840000000000e-05, 6.52635000000000e-05, 1.63902000000000e-06, 0.981817000000000,
                            0.000931234000000000, 0.0111146000000000,
                            4.53602000000000e-05, 0.00140411000000000, 0.0240436000000000, 0.000244897000000000,
                            0.0367757000000000, 0.000300859000000000, 0.897600000000000, 6.24963000000000e-05,
                            0.0221573000000000, 0.0173655000000000,
                            0.000711684000000000, 0.000102446000000000, 0.0393867000000000, 4.42741000000000e-05,
                            9.34642000000000e-06, 0.00128345000000000, 0.958352000000000, 7.34853000000000e-07,
                            0.000104781000000000, 5.02665000000000e-06,
                            7.23479000000000e-06, 0.989291000000000, 0.00411565000000000, 0.00205000000000000,
                            1.78717000000000e-06, 0.000136220000000000, 0.000310446000000000, 0.000203686000000000,
                            0.00323916000000000, 0.000645012000000000,
                            5.14115000000000e-05, 1.84060000000000e-07, 3.29050000000000e-05, 0.000136068000000000,
                            0.913809000000000, 0.00120969000000000, 0.000176452000000000, 0.00500907000000000,
                            0.00206096000000000, 0.0775142000000000,
                            0.0112046000000000, 0.00559835000000000, 0.00162417000000000, 0.614511000000000,
                            5.23965000000000e-05, 0.0557094000000000, 0.000556138000000000, 0.221270000000000,
                            0.0139527000000000, 0.0755208000000000,
                            1.39109000000000e-05, 8.08393000000000e-06, 0.000112781000000000, 0.000706013000000000,
                            0.00380618000000000, 0.00124517000000000, 1.86013000000000e-05, 0.00230427000000000,
                            0.00555493000000000, 0.986230000000000,
                            0.995303000000000, 1.42777000000000e-06, 0.00123188000000000, 0.00165270000000000,
                            2.25622000000000e-08, 0.000414279000000000, 1.93236000000000e-05, 1.72343000000000e-05,
                            0.00135019000000000, 1.03039000000000e-05,
                            0.000173520000000000, 5.38441000000000e-07, 0.000416930000000000, 0.000370661000000000,
                            0.000155947000000000, 0.0198240000000000, 5.49310000000000e-06, 2.24384000000000e-06,
                            0.978228000000000, 0.000823020000000000,
                            3.60207000000000e-05, 6.46853000000000e-05, 0.00111327000000000, 0.0921525000000000,
                            0.000106194000000000, 0.0113428000000000, 1.87324000000000e-05, 0.00188735000000000,
                            0.372861000000000, 0.520417000000000,
                            0.000118603000000000, 2.43526000000000e-08, 0.000114949000000000, 5.25107000000000e-05,
                            0.000114741000000000, 0.0444061000000000, 1.81589000000000e-05, 4.37249000000000e-07,
                            0.954665000000000, 0.000509263000000000,
                            0.000551656000000000, 0.0477557000000000, 0.000805602000000000, 0.0275906000000000,
                            0.102095000000000, 0.0875311000000000, 0.00366723000000000, 0.0663376000000000,
                            0.0130307000000000, 0.650635000000000,
                            7.30152000000000e-06, 4.71824000000000e-07, 0.999716000000000, 2.46981000000000e-05,
                            5.92003000000000e-05, 6.31162000000000e-07, 4.00027000000000e-05, 1.76045000000000e-07,
                            0.000146638000000000, 4.83306000000000e-06,
                            0.000167070000000000, 0.00372082000000000, 0.00105326000000000, 0.00856684000000000,
                            0.0642926000000000, 0.0307146000000000, 0.000333811000000000, 0.265953000000000,
                            0.101119000000000, 0.524079000000000,
                            2.37881000000000e-06, 4.42704000000000e-06, 6.32539000000000e-05, 0.000478354000000000,
                            0.00403681000000000, 7.06146000000000e-05, 5.46080000000000e-06, 0.00231727000000000,
                            0.00362255000000000, 0.989399000000000,
                            8.22539000000000e-06, 0.989282000000000, 0.00116134000000000, 0.00204187000000000,
                            1.03646000000000e-05, 0.00163220000000000, 0.000844928000000000, 9.40383000000000e-05,
                            0.00440830000000000, 0.000516350000000000,
                            0.000130203000000000, 2.11445000000000e-07, 0.000705093000000000, 4.61041000000000e-07,
                            2.36491000000000e-05, 0.000124636000000000, 0.998965000000000, 8.25794000000000e-07,
                            9.08036000000000e-06, 4.07065000000000e-05,
                            0.000873871000000000, 0.00165549000000000, 0.0240797000000000, 0.124667000000000,
                            8.37387000000000e-05, 0.00129478000000000, 0.000125139000000000, 0.00102657000000000,
                            0.746388000000000, 0.0998055000000000,
                            0.000646078000000000, 1.30464000000000e-05, 3.68893000000000e-05, 0.00180701000000000,
                            4.20062000000000e-05, 0.995287000000000, 1.66211000000000e-05, 8.83083000000000e-05,
                            0.00147404000000000, 0.000589231000000000,
                            1.77002000000000e-05, 7.89493000000000e-07, 0.000633247000000000, 7.00624000000000e-05,
                            0.000300937000000000, 0.00126290000000000, 0.990122000000000, 2.33201000000000e-05,
                            0.000264763000000000, 0.00730382000000000,
                            0.00442245000000000, 0.000512370000000000, 0.00720322000000000, 0.00132796000000000,
                            0.00452475000000000, 0.966697000000000, 0.000775848000000000, 0.000833256000000000,
                            0.0134686000000000, 0.000234159000000000,
                            0.00222430000000000, 0.000226844000000000, 0.289806000000000, 1.74713000000000e-05,
                            0.00213643000000000, 0.000968402000000000, 0.608887000000000, 2.85982000000000e-05,
                            0.0913910000000000, 0.00431383000000000,
                            2.15756000000000e-05, 1.60989000000000e-05, 3.89116000000000e-05, 4.94962000000000e-05,
                            0.0172614000000000, 0.000187616000000000, 2.35709000000000e-05, 0.00151838000000000,
                            0.00603511000000000, 0.974848000000000,
                            5.91740000000000e-05, 5.95339000000000e-05, 0.00302412000000000, 0.00115414000000000,
                            0.0215274000000000, 0.00151132000000000, 0.000206865000000000, 0.00423850000000000,
                            0.00660607000000000, 0.961613000000000,
                            0.00196177000000000, 0.000268979000000000, 0.000251580000000000, 0.0128249000000000,
                            0.00238323000000000, 0.957500000000000, 0.000304283000000000, 0.000706826000000000,
                            0.0218105000000000, 0.00198822000000000,
                            0.00670503000000000, 1.33079000000000e-07, 3.71306000000000e-06, 0.000346771000000000,
                            1.80829000000000e-05, 0.550715000000000, 5.46407000000000e-07, 4.55700000000000e-06,
                            0.429607000000000, 0.0125992000000000,
                            0.000374685000000000, 6.37230000000000e-06, 0.00156291000000000, 0.951007000000000,
                            0.000145946000000000, 0.0449485000000000, 3.41256000000000e-06, 2.74617000000000e-07,
                            0.00187196000000000, 7.90463000000000e-05,
                            3.57679000000000e-05, 0.000204612000000000, 0.00182876000000000, 0.00222314000000000,
                            0.000164769000000000, 0.000195267000000000, 8.65658000000000e-06, 0.945858000000000,
                            0.00116045000000000, 0.0483209000000000,
                            1.21276000000000e-05, 9.42152000000000e-07, 1.26813000000000e-06, 1.45556000000000e-05,
                            3.46785000000000e-05, 5.67139000000000e-05, 7.07457000000000e-08, 0.999646000000000,
                            7.17312000000000e-05, 0.000162243000000000,
                            0.0115948000000000, 0.00555136000000000, 0.00493181000000000, 0.121517000000000,
                            0.000149105000000000, 0.274677000000000, 0.00108832000000000, 0.0400549000000000,
                            0.531981000000000, 0.00845372000000000,
                            1.24881000000000e-06, 3.87662000000000e-07, 1.32830000000000e-05, 4.56486000000000e-07,
                            0.998880000000000, 0.000102447000000000, 0.000133484000000000, 0.000429631000000000,
                            0.000132484000000000, 0.000306997000000000,
                            0.201914000000000, 4.08871000000000e-05, 0.00121491000000000, 0.0187559000000000,
                            0.00818567000000000, 0.314508000000000, 0.000577032000000000, 0.000159791000000000,
                            0.0996063000000000, 0.355038000000000,
                            0.00206436000000000, 6.90430000000000e-05, 0.0403082000000000, 0.932399000000000,
                            2.14306000000000e-05, 0.00148689000000000, 7.43397000000000e-05, 0.000427922000000000,
                            0.00136881000000000, 0.0217802000000000,
                            5.02035000000000e-05, 0.976618000000000, 0.00798195000000000, 0.00467246000000000,
                            4.94820000000000e-05, 0.000760145000000000, 0.00306407000000000, 0.000487683000000000,
                            0.00490352000000000, 0.00141247000000000,
                            0.000231869000000000, 4.04947000000000e-05, 0.00968325000000000, 7.43582000000000e-05,
                            0.000104093000000000, 0.000760421000000000, 0.983220000000000, 1.34039000000000e-06,
                            0.00537358000000000, 0.000510831000000000,
                            0.00960489000000000, 0.0232651000000000, 0.338912000000000, 0.102841000000000,
                            2.62535000000000e-05, 0.00343308000000000, 0.000129758000000000, 7.50040000000000e-06,
                            0.521667000000000, 0.000113968000000000,
                            0.989210000000000, 3.51769000000000e-08, 0.000226473000000000, 0.000117843000000000,
                            4.72952000000000e-06, 0.0100096000000000, 0.000353658000000000, 3.08442000000000e-05,
                            1.77269000000000e-05, 2.94339000000000e-05,
                            0.000143537000000000, 4.99079000000000e-06, 0.997983000000000, 0.000634036000000000,
                            5.62691000000000e-06, 4.39366000000000e-06, 7.11981000000000e-05, 5.82157000000000e-05,
                            0.00105326000000000, 4.16448000000000e-05,
                            0.977485000000000, 5.83786000000000e-09, 0.000109779000000000, 2.65213000000000e-06,
                            1.77024000000000e-06, 0.0222394000000000, 0.000108454000000000, 7.97645000000000e-06,
                            3.89457000000000e-05, 5.55484000000000e-06,
                            0.985068000000000, 6.20530000000000e-08, 1.88494000000000e-06, 0.00911128000000000,
                            9.97120000000000e-06, 0.00438493000000000, 6.75142000000000e-08, 0.000694883000000000,
                            0.000220350000000000, 0.000508493000000000,
                            0.000162112000000000, 7.04062000000000e-07, 0.000180622000000000, 1.86725000000000e-07,
                            1.02871000000000e-05, 5.00169000000000e-05, 0.999571000000000, 1.18103000000000e-06,
                            1.23752000000000e-05, 1.19438000000000e-05,
                            5.33434000000000e-07, 2.68295000000000e-07, 1.76414000000000e-05, 0.000181581000000000,
                            0.00174473000000000, 3.36281000000000e-05, 4.73343000000000e-06, 0.00111903000000000,
                            0.000625809000000000, 0.996272000000000,
                            3.15006000000000e-05, 2.75000000000000e-05, 2.67615000000000e-05, 0.00334326000000000,
                            0.00333258000000000, 0.000917740000000000, 3.02575000000000e-06, 0.0273153000000000,
                            0.00267050000000000, 0.962332000000000,
                            0.000470044000000000, 0.00235890000000000, 0.0485873000000000, 0.000445396000000000,
                            0.000849510000000000, 0.00746206000000000, 0.00459535000000000, 2.21023000000000e-05,
                            0.934775000000000, 0.000434126000000000,
                            0.899563000000000, 1.55832000000000e-08, 5.95698000000000e-07, 4.43115000000000e-06,
                            6.40817000000000e-08, 0.100389000000000, 1.15228000000000e-05, 3.50446000000000e-06,
                            2.74294000000000e-05, 7.54460000000000e-07,
                            0.000520242000000000, 0.0125074000000000, 0.891198000000000, 0.0259093000000000,
                            0.00220700000000000, 0.00336898000000000, 0.0319299000000000, 0.0105408000000000,
                            0.0109746000000000, 0.0108439000000000,
                            7.33080000000000e-06, 0.986963000000000, 0.00689759000000000, 0.00120031000000000,
                            0.000147488000000000, 7.67711000000000e-05, 3.88767000000000e-05, 0.000628545000000000,
                            0.00388149000000000, 0.000158726000000000,
                            0.00481886000000000, 0.000420836000000000, 0.000450564000000000, 0.972137000000000,
                            2.93947000000000e-06, 0.00184472000000000, 5.88755000000000e-07, 0.00400613000000000,
                            0.00191288000000000, 0.0144060000000000,
                            6.79794000000000e-06, 0.975237000000000, 0.0186930000000000, 0.000611081000000000,
                            9.03029000000000e-06, 4.07810000000000e-05, 0.000104389000000000, 4.77827000000000e-05,
                            0.00518167000000000, 6.87660000000000e-05,
                            0.000166844000000000, 8.05188000000000e-06, 0.000358698000000000, 4.47869000000000e-05,
                            0.154325000000000, 0.00122146000000000, 0.000871632000000000, 0.000108081000000000,
                            0.0991335000000000, 0.743762000000000,
                            0.000134101000000000, 2.62811000000000e-06, 0.00359123000000000, 0.993423000000000,
                            1.60284000000000e-05, 0.00157235000000000, 3.26158000000000e-05, 3.26751000000000e-07,
                            0.00110994000000000, 0.000117536000000000,
                            0.782556000000000, 1.65894000000000e-05, 0.00560939000000000, 0.00364386000000000,
                            0.000652178000000000, 0.0191118000000000, 0.0227429000000000, 0.00691778000000000,
                            0.0846038000000000, 0.0741453000000000,
                            7.74377000000000e-05, 0.000160078000000000, 0.00323115000000000, 0.00616239000000000,
                            0.0225106000000000, 0.000140589000000000, 0.000478086000000000, 0.000324138000000000,
                            0.00106905000000000, 0.965847000000000,
                            0.00653762000000000, 4.37634000000000e-05, 0.0199877000000000, 0.000490632000000000,
                            0.646829000000000, 0.000580940000000000, 0.0150879000000000, 0.0225014000000000,
                            0.0371927000000000, 0.250748000000000,
                            7.93678000000000e-06, 0.975942000000000, 0.0116914000000000, 0.00101921000000000,
                            7.00190000000000e-06, 4.10228000000000e-05, 2.48576000000000e-05, 3.37652000000000e-05,
                            0.0111404000000000, 9.22334000000000e-05,
                            0.000136375000000000, 0.00702270000000000, 0.0137462000000000, 0.0408365000000000,
                            0.000159755000000000, 0.00460231000000000, 8.54830000000000e-05, 0.00112632000000000,
                            0.906027000000000, 0.0262570000000000,
                            0.914064000000000, 2.52326000000000e-07, 6.39782000000000e-05, 4.16162000000000e-05,
                            5.45744000000000e-06, 0.0848460000000000, 0.000663548000000000, 0.000115606000000000,
                            0.000143529000000000, 5.55616000000000e-05,
                            6.68172000000000e-05, 1.26904000000000e-07, 0.000744435000000000, 1.10181000000000e-05,
                            0.834906000000000, 3.40728000000000e-06, 0.00160575000000000, 3.06412000000000e-05,
                            0.00183982000000000, 0.160792000000000,
                            0.519499000000000, 7.07315000000000e-08, 0.000646355000000000, 0.00468903000000000,
                            0.000893415000000000, 0.00673781000000000, 2.10108000000000e-06, 0.438626000000000,
                            0.000317806000000000, 0.0285886000000000,
                            0.928000000000000, 1.86812000000000e-05, 0.0378427000000000, 0.0135150000000000,
                            3.70865000000000e-08, 0.00240431000000000, 3.08054000000000e-06, 0.0138033000000000,
                            0.00374939000000000, 0.000663407000000000,
                            3.15160000000000e-05, 0.962024000000000, 0.0182258000000000, 0.00695550000000000,
                            1.24918000000000e-05, 0.000229818000000000, 0.000606943000000000, 0.000357375000000000,
                            0.0108239000000000, 0.000732721000000000,
                            3.16927000000000e-07, 4.12426000000000e-07, 5.91465000000000e-06, 0.000190763000000000,
                            8.33332000000000e-07, 1.07555000000000e-06, 3.66414000000000e-08, 0.998780000000000,
                            3.91365000000000e-06, 0.00101696000000000,
                            2.58610000000000e-05, 7.41454000000000e-06, 0.994423000000000, 2.06940000000000e-05,
                            2.06895000000000e-06, 3.86067000000000e-05, 7.90282000000000e-05, 7.40906000000000e-07,
                            0.00392292000000000, 0.00147949000000000,
                            1.44903000000000e-05, 3.18908000000000e-05, 0.00217409000000000, 0.000622050000000000,
                            8.39756000000000e-06, 1.07593000000000e-05, 1.46355000000000e-06, 0.977276000000000,
                            0.00229960000000000, 0.0175615000000000,
                            0.305119000000000, 0.00153466000000000, 0.127419000000000, 0.511876000000000,
                            1.02288000000000e-06, 0.0338536000000000, 0.00271941000000000, 0.00561095000000000,
                            0.0118071000000000, 5.87673000000000e-05,
                            3.10536000000000e-05, 1.22891000000000e-05, 6.75233000000000e-07, 0.000660133000000000,
                            0.0298113000000000, 0.000609019000000000, 1.45983000000000e-05, 0.896517000000000,
                            0.00525132000000000, 0.0670927000000000,
                            3.46799000000000e-06, 5.43202000000000e-05, 2.77490000000000e-05, 1.17190000000000e-05,
                            0.991494000000000, 0.000530903000000000, 7.61063000000000e-05, 0.000218418000000000,
                            0.000954260000000000, 0.00662916000000000,
                            0.735845000000000, 6.07378000000000e-06, 0.000665765000000000, 0.0304755000000000,
                            4.45075000000000e-07, 0.00751246000000000, 3.78720000000000e-07, 0.0779272000000000,
                            0.0109176000000000, 0.136649000000000,
                            0.00229848000000000, 2.32295000000000e-05, 6.04200000000000e-05, 0.00226311000000000,
                            2.29051000000000e-07, 0.991531000000000, 1.47548000000000e-06, 4.05568000000000e-07,
                            0.00381772000000000, 4.48740000000000e-06,
                            0.000363523000000000, 0.000127560000000000, 0.0185042000000000, 0.00203342000000000,
                            0.0123249000000000, 0.00665605000000000, 0.000667000000000000, 0.000124928000000000,
                            0.957725000000000, 0.00147325000000000,
                            2.78187000000000e-05, 0.00621583000000000, 0.0679620000000000, 0.000104895000000000,
                            4.04628000000000e-05, 0.00150580000000000, 0.918566000000000, 1.69866000000000e-05,
                            0.00534733000000000, 0.000212669000000000,
                            1.49278000000000e-05, 3.65305000000000e-05, 6.15190000000000e-05, 0.000816370000000000,
                            0.00318019000000000, 0.000469905000000000, 1.08844000000000e-05, 0.00139933000000000,
                            0.00123666000000000, 0.992774000000000,
                            0.000171148000000000, 4.01692000000000e-05, 0.000127309000000000, 0.000274554000000000,
                            0.000163761000000000, 0.000977138000000000, 1.12107000000000e-05, 0.941013000000000,
                            0.00127847000000000, 0.0559429000000000,};


            float label[] = {8
                    , 7
                    , 1
                    , 1
                    , 3
                    , 7
                    , 1
                    , 1
                    , 7
                    , 6
                    , 1
                    , 7
                    , 2
                    , 3
                    , 0
                    , 3
                    , 3
                    , 6
                    , 8
                    , 5
                    , 4
                    , 6
                    , 0
                    , 2
                    , 8
                    , 8
                    , 6
                    , 7
                    , 6
                    , 6
                    , 1
                    , 4
                    , 3
                    , 9
                    , 0
                    , 8
                    , 8
                    , 8
                    , 9
                    , 2
                    , 4
                    , 9
                    , 1
                    , 6
                    , 8
                    , 5
                    , 6
                    , 5
                    , 8
                    , 9
                    , 9
                    , 5
                    , 8
                    , 3
                    , 7
                    , 7
                    , 8
                    , 4
                    , 5
                    , 3
                    , 1
                    , 6
                    , 2
                    , 0
                    , 2
                    , 0
                    , 0
                    , 6
                    , 9
                    , 9
                    , 8
                    , 0
                    , 2
                    , 1
                    , 3
                    , 1
                    , 4
                    , 3
                    , 0
                    , 2
                    , 4
                    , 1
                    , 8
                    , 0
                    , 4
                    , 0
                    , 0
                    , 1
                    , 7
                    , 2
                    , 7
                    , 3
                    , 7
                    , 4
                    , 0
                    , 5
                    , 8
                    , 6
                    , 9
                    , 7};

            float res = 0;
            Context context = Context::Test();
            Shape dataShape = ShapeN(100,10);
            Shape labelShape = ShapeN(100);
            std::vector<void *> inputs;
            inputs.push_back(data);
            inputs.push_back(label);
            std::vector<Shape *> inShape;
            inShape.push_back(&dataShape);
            inShape.push_back(&labelShape);
            OpPtr pro = Registry::Global()->GetOp("accuracy");

            std::map<std::string, Any> params;

            Shape out;
            Operator *op = pro->CreateOperator(context,  &inShape, &out, params);
            op->SetData(&inputs, &res);
            op->AsyncRun();
            PrintMat(&res, 1, 1, "acc");
        }


        TEST_F(OpTest, FlattenOp) {
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
            float res[25*3] = {0};

            OpPtr pro = Registry::Global()->GetOp("flatten");

            Context context = Context::Test();

            std::vector<void *> inputs;
            inputs.push_back(data);
            Shape dataShape = ShapeN(1, 3, 5, 5);
            std::vector<Shape *> inShape;
            inShape.push_back(&dataShape);
            Shape out;
            std::map<std::string, Any> params;
            Operator *op = pro->CreateOperator(context, &inShape, &out, params);
            op->SetData(&inputs, res);
            op->AsyncRun();
            int dim = out.Size();

            PrintMat(res, 15, 5, "FlattenOp_test_result");
            checkArrayEqual<float>(data, res, dim);
        }


        TEST_F(OpTest, FlattenGradOp) {
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
            float res[25*3] = {0};

            OpPtr pro = Registry::Global()->GetOp("grad_flatten");

            Context context = Context::Test();

            std::vector<void *> inputs;
            inputs.push_back(data);
            inputs.push_back(nullptr);
            inputs.push_back(nullptr);
            Shape preShape = ShapeN(1, 75);
            Shape dataShape = ShapeN(1, 3, 5, 5);
            std::vector<Shape *> inShape;
            inShape.push_back(&preShape);
            inShape.push_back(&preShape);
            inShape.push_back(&dataShape);
            Shape out;
            std::map<std::string, Any> params;
            Operator *op = pro->CreateOperator(context, &inShape, &out, params);
            op->SetData(&inputs, res);
            op->AsyncRun();
            int dim = out.Size();

            PrintMat(res, 15, 5, "FlattenOp_test_result");
            checkArrayEqual<float>(data, res, dim);
        }
    }
}