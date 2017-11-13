//
// Created by Jarlene on 2017/7/23.
//

#include <matrix/include/api/ConstantSymbol.h>
#include <matrix/include/executor/Executor.h>

using namespace matrix;

int main() {


    float a[] = {1, 2, 3,
                 4, 5, 6};
    float b[] = {2, 3,
                 4, 5,
                 6, 7};

    auto s1 = ShapeN(2, 3);
    auto as = ConstantSymbol::Create<float>("a", a, s1);
    auto s2 = ShapeN(3, 2);
    auto bs = ConstantSymbol::Create<float>("b", b, s2);

    auto cs = as * bs;

    Context context;
    context.type = kFloat;
    context.phase = TEST;
    context.mode = kCpu;

    auto executor = std::make_shared<Executor>(cs, context);
    executor->runAsync();

    cs.PrintMatrix<float>();


    return 0;
}