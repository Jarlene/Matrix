//
// Created by Jarlene on 2017/7/23.
//

#include <matrix/include/utils/Math.h>
#include <matrix/include/utils/Logger.h>
#include <matrix/include/utils/Time.h>
#include <iostream>


using namespace matrix;

int main() {

    long start, end;

    float a[] = {1, 2, 3, 4, 5, 6};
    float b[] = {2, 3, 4, 5, 6, 7};
    float c[] = {0, 0, 0, 0, 0, 0};
    CPUGemm(NoTrans, NoTrans, 2, 3, 2, 1.0f, a, b, 0.0f, c);
    for (int i = 0; i < 6; ++i) {
       Logger::Global()->Info("%d\n",c[i]);
    }

    Logger::Global()->Info("the core count is %d \n", omp_get_max_threads());
    start = getCurrentTime();
#pragma omp parallel for
    for (int i = 0; i < 10000; i++) {
        Logger::Global()->Info(" the index is %d, the thread is %d \n", i, omp_get_thread_num());
    }
    end = getCurrentTime();

    Logger::Global()->Info("计算耗时为：%d \n", end-start);

    omp_set_num_threads(CPU_CORES);
#pragma omp parallel for
    for (int i = 0; i < 6; i++)
        Logger::Global()->Info("i = %d, I am Thread %d\n", i, omp_get_thread_num());
    return 1;
}