//
// Created by Jarlene on 2017/7/23.
//

#include <matrix/include/utils/Math.h>
#include <iostream>

using namespace matrix;

long getCurrentTime() {
    struct timeval tv;
    gettimeofday(&tv,NULL);
    return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

int main() {

    long start, end;

    float a[] = {1, 2, 3, 4, 5, 6};
    float b[] = {2, 3, 4, 5, 6, 7};
    float c[] = {0, 0, 0, 0, 0, 0};
    CPUGemm(NoTrans, NoTrans, 2, 3, 2, 1.0f, a, b, 0.0f, c);
    for (int i = 0; i < 6; ++i) {
        std::cout<< c[i] << std::endl;
    }


    std::cout<< "the core count is " << omp_get_max_threads() << std::endl;
    start = getCurrentTime();
#pragma omp parallel for
    for (int i = 0; i < 10000; i++) {
        std::cout << " the index is " << i << ", the thread is " << omp_get_thread_num() << std::endl;
    }
    end = getCurrentTime();

    std::cout<<"计算耗时为："<<end -start<<std::endl;

    omp_set_num_threads(CPU_CORES);
#pragma omp parallel for
    for (int i = 0; i < 6; i++)
        printf("i = %d, I am Thread %d\n", i, omp_get_thread_num());


    std::cout << "the max threads is :" << omp_get_num_threads() << std::endl;

    return 1;
}