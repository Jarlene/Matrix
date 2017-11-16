//
// Created by Jarlene on 2017/11/15.
//

#include <matrix/include/api/ConstantSymbol.h>
#include <matrix/include/optimizer/SGDOptimizer.h>
#include <matrix/include/executor/Executor.h>
#include <matrix/include/api/VariableSymbol.h>

using namespace matrix;


int main(int argc, char *argv[]) {

    float a = 3;
    auto as = ConstantSymbol::Create("a", &a, ShapeN(1));
    auto bs = VariableSymbol::Create("b", ShapeN(1));

    auto cs = as + bs;

//    auto ds = Symbol("activation")
//            .SetInput("cs", cs)
//            .SetParam("type", kSigmoid)
//            .Build("act1");
//
//    auto es = Symbol("activation")
//            .SetInput("cs", cs)
//            .SetParam("type", kRelu)
//            .Build("act2");

    auto fs = cs * cs;

    auto loss = Symbol("loss")
            .SetInput("logistic", fs)
            .SetInput("y", as)
            .SetParam("type", kMSE)
            .Build("loss");

    Context context;
    context.type = kFloat;
    context.phase = TRAIN;
    context.mode = kCpu;

    auto opt = new SGDOptimizer(0.001f);
    auto executor = std::make_shared<Executor>(loss, context, opt);
    for (int i = 0; i < 100; ++i) {
        executor->runSync();
        loss.PrintMatrix<float>();
        if ((1+i)%100 == 0) {
            bs.PrintMatrix<float>();
        }
    }
    return 0;
}