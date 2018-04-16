//
// Created by Jarlene on 2017/12/11.
//

#include <matrix/include/utils/Logger.h>
#include <matrix/include/api/Symbol.h>
#include <matrix/include/api/PlaceHolderSymbol.h>
#include <matrix/include/executor/Executor.h>
#include <matrix/include/utils/Time.h>
#include <matrix/include/optimizer/SGDOptimizer.h>
#include <matrix/include/utils/Dispatcher.h>
using namespace matrix;

Symbol logistic(Symbol &input, int unit, int hideNum, int classNum) {
    auto y1 = Symbol("lstm")
            .SetInput("data", input)
            .SetParam("hide_num", unit)
            .SetParam("with_bias", false)
            .Build("y1");

    auto y2 = Symbol("fullConnected")
            .SetInput("y1", y1)
            .SetParam("hide_num", hideNum)
            .SetParam("with_bias", true)
            .SetParam("activation_type", kRelu)
            .Build("y2");


    auto y3 = Symbol("fullConnected")
            .SetInput("y2", y2)
            .SetParam("hide_num", classNum)
            .SetParam("with_bias", true)
            .Build("y3");

    auto out = Symbol("output")
            .SetInput("y3", y3)
            .SetParam("type", kSoftmax)
            .Build("out");

    return out;
}

int main(int argc, char *argv[]) {

    // hyper-parameters
    const int epochSize = 2000;

    const int batch_size = 128;
    const int maxLength = 100;
    const int embedding = 200;

    const int unit = 64;
    const int hideNum = 32;
    const int classNum = 10;

    // train data
    Shape seqShape =  ShapeN(batch_size, maxLength, embedding);
    Shape labelShape = ShapeN(batch_size);
    auto data = PlaceHolderSymbol::Create("data", seqShape);
    auto label = PlaceHolderSymbol::Create("label", labelShape);

    float* seqData = static_cast<float *>(malloc(sizeof(float) * seqShape.Size()));
    float* labelData = static_cast<float *>(malloc(sizeof(float) * labelShape.Size()));

    // classifier
    auto logist = logistic(data, unit, hideNum, classNum);

    // loss
    auto loss = Symbol("loss")
            .SetInput("logistic", logist)
            .SetInput("label", label)
            .SetParam("type", kCrossEntropy)
            .Build("loss");

    // accuracy
    auto acc = Symbol("accuracy")
            .SetInput("logistic", logist)
            .SetInput("label", label)
            .Build("acc");

    // run context
    Context context = Context::Default();
    // optimizer sgd
    auto opt = new SGDOptimizer(0.01f);

    // graph executor
    auto executor = make<Executor>(loss, context, opt);

    // training
    for (int i = 0; i < epochSize; ++i) {

        data.Fill(seqData);
        label.Fill(labelData);
        long start = getCurrentTime();
        executor->train(&acc);
        executor->update();
        long end = getCurrentTime();
        std::cout << "the epoch[" << (i + 1) << "] take time: " << end - start << " ms" << std::endl;
        loss.PrintMatrix();
        acc.PrintMatrix();
    }

    return 0;
}