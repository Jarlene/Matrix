//
// Created by Jarlene on 2017/7/23.
//

#include <matrix/include/api/ConstantSymbol.h>
#include <matrix/include/api/VariableSymbol.h>
#include <matrix/include/executor/Executor.h>
#include <matrix/include/optimizer/SGDOptimizer.h>
#include <matrix/include/utils/Time.h>

using namespace matrix;

int main() {


    float a[] = {2};

    auto s1 = ShapeN(1,1);
    auto as = ConstantSymbol::Create<float>("as", a, s1);
    auto bs = VariableSymbol::Create("bs", ShapeN(1,1));
    auto ds = as + bs;
    auto es = ds * ds - bs;
    Context context = Context::Default();
    auto opt = new SGDOptimizer(0.01f);
    auto executor = std::make_shared<Executor>(es, context, opt);
    long start = getCurrentTime();
    for (int i = 0; i < 500; ++i) {
        executor->train();
        executor->update();
        bs.PrintMatrix();
    }
    long end = getCurrentTime();
    std::cout << "spend time is "<< (end-start) << "ms" << std::endl;
    return 0;
}