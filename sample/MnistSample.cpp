//
// Created by Jarlene on 2017/7/18.
//
#include <matrix/include/utils/Logger.h>
#include <matrix/include/api/Symbol.h>
#include <matrix/include/optimizer/MomentOptimizer.h>
#include <matrix/include/api/PlaceHolderSymbol.h>
#include <matrix/include/executor/Executor.h>
#include <matrix/include/models/models.h>
#include <matrix/include/utils/Time.h>
#include <matrix/include/optimizer/SGDOptimizer.h>
#include "include/MnistDataSet.h"


using namespace matrix;

const string trainImagePath = "../../data/train-images-idx3-ubyte";
const string trainLabelPath = "../../data/train-labels-idx1-ubyte";
const string testImagePath = "../../data/t10k-images-idx3-ubyte";
const string testLabelPath = "../../data/t10k-labels-idx1-ubyte";


void AlexNet(int batchSize, int class_num, int epochSize) {
    Shape imageShape = ShapeN(batchSize,  1, 228, 228);
    Shape labelShape = ShapeN(batchSize);

    float* imageData = static_cast<float *>(malloc(sizeof(float) * imageShape.Size()));
    float* labelData = static_cast<float *>(malloc(sizeof(float) * labelShape.Size()));

    auto image = PlaceHolderSymbol::Create("image", imageShape);
    auto label = PlaceHolderSymbol::Create("label", labelShape);

    auto train = AlexSymbol(image, class_num);

    auto loss = Symbol("loss")
            .SetInput("logistic", train)
            .SetInput("y", label)
            .SetParam("type", kCrossEntropy)
            .Build("loss");

    auto acc = Symbol("accuracy")
            .SetInput("logistic", train)
            .SetInput("y", label)
            .Build("acc");

    Context context = Context::Default();
    auto opt = new SGDOptimizer(0.01f);

    auto executor = std::make_shared<Executor>(loss, context, opt);
    for (int i = 0; i < epochSize; ++i) {
        image.Fill(imageData);
        label.Fill(labelData);
        long start = getCurrentTime();
        executor->train(&acc);
        executor->update();
        long end = getCurrentTime();
        std::cout << "the epoch[" << (i + 1) << "] take time: " << end - start << " ms" << std::endl;
        loss.PrintMatrix();
        acc.PrintMatrix();
    }

    free(imageData);
    free(labelData);
}

void Mnist(int batchSize, int hideNum, int class_num, int epochSize, bool isConv) {
    Shape labelShape = ShapeN(batchSize);
    Shape imageShape;
    if (isConv) {
        imageShape.reShape(ShapeN(batchSize,  1, 28, 28));
    } else {
        imageShape.reShape(ShapeN(batchSize, 784));
    }

    auto image = PlaceHolderSymbol::Create("image", imageShape);
    auto label = PlaceHolderSymbol::Create("label", labelShape);

    float* imageData = static_cast<float *>(malloc(sizeof(float) * imageShape.Size()));
    float* labelData = static_cast<float *>(malloc(sizeof(float) * labelShape.Size()));
    Symbol logistic;
    if (isConv) {
        logistic = MnistConv(image, hideNum, class_num);
    } else {
        logistic = MnistFc(image, hideNum, class_num);
    }

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
    auto opt = new MomentOptimizer;
    MnistDataSet trainSet(trainImagePath, trainLabelPath);
    MnistDataSet testSet(testImagePath, testLabelPath);
    auto executor = std::make_shared<Executor>(loss, context, opt);
    for (int i = 0; i < epochSize; ++i) {
        trainSet.getMiniBatch(batchSize, imageData, labelData);
        image.Fill(imageData);
        label.Fill(labelData);
        long start = getCurrentTime();
        executor->train(&acc);
        executor->update();
        long end = getCurrentTime();
        std::cout << "the epoch[" << (i + 1) << "] take time: " << end - start << " ms" << std::endl;
        loss.PrintMatrix();
        acc.PrintMatrix();
        if ((i + 1) % (trainSet.getNumberOfImages()/batchSize) == 0) {
            int total_run_data = 0;
            int test_correct = 0;
            int total = testSet.getNumberOfImages();
            for (int j = 0; j < total; j += batchSize) {
                testSet.getMiniBatch(batchSize, imageData, labelData);
                image.Fill(imageData);
                label.Fill(labelData);
                total_run_data += batchSize;
                float *cnt = static_cast<float *>(executor->evaluating(&acc));
                test_correct += int((*cnt * batchSize) + 0.5);
            }
            std::cout << "test correct rate: " << test_correct * 1.0f / total << std::endl;
        }
    }

    free(imageData);
    free(labelData);
}

int main() {
    const int batchSize = 300;
    const int epochSize = 60000/batchSize * 10;
    const int classNum = 10;
    const int hideNum = 128;
    Mnist(batchSize, hideNum, classNum, epochSize, false);
    return 0;
}