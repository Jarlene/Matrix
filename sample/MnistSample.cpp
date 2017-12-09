//
// Created by Jarlene on 2017/7/18.
//
#include <matrix/include/utils/Logger.h>
#include <matrix/include/api/Symbol.h>
#include <matrix/include/optimizer/SGDOptimizer.h>
#include <matrix/include/api/PlaceHolderSymbol.h>
#include <matrix/include/executor/Executor.h>
#include <matrix/include/utils/Time.h>
#include "include/MnistDataSet.h"


using namespace matrix;

const string trainImagePath = "../../data/train-images-idx3-ubyte";
const string trainLabelPath = "../../data/train-labels-idx1-ubyte";
const string testImagePath = "../../data/t10k-images-idx3-ubyte";
const string testLabelPath = "../../data/t10k-labels-idx1-ubyte";


Symbol LogisticRegression(const Symbol &input, int hideNum, int classNum) {

    auto y1 = Symbol("fullConnected")
            .SetInput("data", input)
            .SetParam("hide_num", hideNum)
            .SetParam("with_bias", true)
            .Build("y1");

    auto act1 = Symbol("activation")
            .SetInput("y1", y1)
            .SetParam("type", kRelu)
            .Build("act1");

    auto y2 = Symbol("fullConnected")
            .SetInput("act1", act1)
            .SetParam("hide_num", classNum)
            .SetParam("with_bias", true)
            .Build("y2");

    auto out = Symbol("output")
            .SetInput("y2", y2)
            .SetParam("type", kSoftmax)
            .Build("out");

    return out;
}


Symbol Connvolution(const Symbol &input, int hideNum, int classNum) {

    auto conv1 = Symbol("convolution")
            .SetInput("data", input)
            .SetParam("filter", ShapeN(5, 5))
            .SetParam("with_bias", true)
            .SetParam("padding", ShapeN(0,0))
            .SetParam("stride", ShapeN(1,1))
            .SetParam("dilate", ShapeN(1,1))
            .SetParam("filter_num", 20)
            .SetParam("group", 1)
            .Build("conv1");

    auto act = Symbol("activation")
            .SetInput("conv1", conv1)
            .SetParam("type", ActType::kRelu)
            .Build("act");

    auto pool1 = Symbol("pooling")
            .SetInput("act", act)
            .SetParam("filter", ShapeN(2,2))
            .SetParam("stride", ShapeN(2,2))
            .SetParam("type", PoolType::kMax)
            .Build("pool1");


    auto conv2 = Symbol("convolution")
            .SetInput("pool1", pool1)
            .SetParam("filter",  ShapeN(5, 5))
            .SetParam("with_bias", true)
            .SetParam("padding", ShapeN(0,0))
            .SetParam("stride", ShapeN(1,1))
            .SetParam("dilate", ShapeN(1,1))
            .SetParam("filter_num", 40)
            .SetParam("group", 1)
            .Build("conv2");

    auto act2 = Symbol("activation")
            .SetInput("conv2", conv2)
            .SetParam("type", ActType::kRelu)
            .Build("act2");

    auto pool2 = Symbol("pooling")
            .SetInput("act2", act2)
            .SetParam("filter", ShapeN(2,2))
            .SetParam("stride", ShapeN(2,2))
            .SetParam("type", PoolType::kMax)
            .Build("pool2");

    auto flatten = Symbol("flatten")
            .SetInput("pool2", pool2)
            .Build("flatten");

    auto fc = Symbol("fullConnected")
            .SetInput("flatten", flatten)
            .SetParam("hide_num", hideNum)
            .SetParam("with_bias", true)
            .Build("fc1");

    auto act3 = Symbol("activation")
            .SetInput("fc", fc)
            .SetParam("type", ActType::kRelu)
            .Build("act3");

    auto fc2 = Symbol("fullConnected")
            .SetInput("act3", act3)
            .SetParam("hide_num", classNum)
            .SetParam("with_bias", true)
            .Build("fc2");

    auto out = Symbol("output")
            .SetInput("fully", fc2)
            .SetParam("type", kSoftmax)
            .Build("out");

    return out;
}



int main() {
    const int batchSize = 100;
    const int epochSize = 5000;
    const int classNum = 10;
    const int hideNum = 128;

    Shape imageShape = ShapeN(batchSize,  1, 28, 28);
    Shape labelShape = ShapeN(batchSize);
    auto image = PlaceHolderSymbol::Create("image", imageShape);
    auto label = PlaceHolderSymbol::Create("label", labelShape);

    float* imageData = static_cast<float *>(malloc(sizeof(float) * imageShape.Size()));
    float* labelData = static_cast<float *>(malloc(sizeof(float) * labelShape.Size()));


    auto logistic = Connvolution(image, hideNum, classNum);

//    auto logistic = LogisticRegression(image, hideNum, classNum);

    auto loss = Symbol("loss")
            .SetInput("logistic", logistic)
            .SetInput("y", label)
            .SetParam("type", kCrossEntropy)
            .Build("loss");

    auto acc = Symbol("accuracy")
            .SetInput("logistic", logistic)
            .SetInput("y", label)
            .Build("acc");


    Context context = Context::Default();
    auto opt = new SGDOptimizer(0.5f/batchSize);
    MnistDataSet trainSet(trainImagePath, trainLabelPath);
    MnistDataSet testSet(testImagePath, testLabelPath);
    auto executor = std::make_shared<Executor>(loss, context, opt);
    for (int i = 0; i < epochSize; ++i) {
        trainSet.getMiniBatch(batchSize, imageData, labelData);
        image.Fill(imageData);
        label.Fill(labelData);
        executor->train(&acc);
        executor->update();
        if ((i + 1) % 100 == 0) {
            loss.PrintMatrix();
            acc.PrintMatrix();
        }
        if ((i + 1) % 1000 == 0) {
            int total_run_data = 0;
            int test_correct = 0;
            int total = testSet.getNumberOfImages();
            for (int j = 0; j < total; j += batchSize) {
                testSet.getMiniBatch(batchSize, imageData, labelData);
                image.Fill(imageData);
                label.Fill(labelData);
                total_run_data += batchSize;
                float *cnt = static_cast<float *>(executor->evaluating(&acc));
                test_correct += *cnt * batchSize;
            }
            std::cout << "correct rate: " << test_correct * 1.0f / total << std::endl;
        }
    }

    free(imageData);
    free(labelData);

    return 0;
}