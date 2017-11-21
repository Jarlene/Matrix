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
    auto as = ConstantSymbol::Create("a", &a, ShapeN(1, 1));
    auto bs = VariableSymbol::Create("b", ShapeN(1, 1));

    float b = 1;
    auto ts = ConstantSymbol::Create("ts", &b, ShapeN(1, 1));

    auto cs = as + bs;

    auto ds = Symbol("activation")
            .SetInput("cs", cs)
            .SetParam("type", kSigmoid)
            .Build("act1");

    auto es = Symbol("activation")
            .SetInput("cs", cs)
            .SetParam("type", kSigmoid)
            .Build("act2");

    auto fs = ds * es;


    Context context;
    context.type = kFloat;
    context.phase = TRAIN;
    context.mode = kCpu;

    auto opt = new SGDOptimizer(0.001f);
    auto executor = std::make_shared<Executor>(fs, context, opt);
    for (int i = 0; i < 5000; ++i) {
        executor->train();
        fs.PrintMatrix();
        if ((1 + i) % 100 == 0) {
            bs.PrintMatrix();
        }
    }
    return 0;
}