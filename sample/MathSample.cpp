//
// Created by Jarlene on 2017/7/23.
//

#include <matrix/include/api/ConstantSymbol.h>
#include <matrix/include/api/VariableSymbol.h>
#include <matrix/include/executor/Executor.h>
#include <matrix/include/optimizer/SGDOptimizer.h>
#include <matrix/include/optimizer/MomentOptimizer.h>
#include <matrix/include/utils/Time.h>
#include <matrix/include/utils/Dispatcher.h>

using namespace matrix;

int main() {


    float a[] = {2};

    auto s1 = ShapeN(1,1);
    auto as = ConstantSymbol::Create<float>("as", a, s1);
    auto bs = VariableSymbol::Create("bs", ShapeN(1,1));
    auto ds = as + bs;
    auto es = ds * ds - bs;
    auto fs = es - ds;
    Context context = Context::Default();
    auto opt = new MomentOptimizer;
    auto executor = make<Executor>(fs, context, opt);

    for (int i = 0; i < 200; ++i) {
        long start = getCurrentTime();
        executor->train();
        executor->update();
        long end = getCurrentTime();
        std::cout << "the epoch[" << (i + 1) << "] take time: " << end - start << " ms" << std::endl;
        bs.PrintMatrix();
    }
    return 0;
}