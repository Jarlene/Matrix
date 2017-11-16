//
// Created by Jarlene on 2017/7/23.
//

#include <matrix/include/api/ConstantSymbol.h>
#include <matrix/include/api/VariableSymbol.h>
#include <matrix/include/executor/Executor.h>
#include <matrix/include/optimizer/SGDOptimizer.h>

using namespace matrix;

int main() {


    float a[] = {2};

    auto s1 = ShapeN(1,1);
    auto as = ConstantSymbol::Create<float>("as", a, s1);
    auto bs = VariableSymbol::Create("bs", ShapeN(1,1));
    auto cs = as + bs;

    auto ds = cs * cs + bs;

    Context context;
    context.type = kFloat;
    context.phase = TRAIN;
    context.mode = kCpu;

    auto opt = new SGDOptimizer(0.1f);
    auto executor = std::make_shared<Executor>(ds, context, opt);
    for (int i = 0; i < 10; ++i) {
        executor->runAsync();
        bs.PrintMatrix<float>();
    }
    return 0;
}