//
// Created by Jarlene on 2017/7/23.
//

#include <matrix/include/utils/Math.h>
#include <iostream>

using namespace matrix;

int main() {

    float a[] = {1, 2, 3, 4, 5, 6};
    float b[] = {2, 3, 4, 5, 6, 7};
    float c[] = {0, 0, 0, 0, 0, 0};
    CPUGemm(NoTrans, NoTrans, 2, 3, 2, 1.0f, a, b, 0.0f, c);
    for (int i = 0; i < 6; ++i) {
        std::cout<< c[i] << std::endl;
    }
    return 1;
}