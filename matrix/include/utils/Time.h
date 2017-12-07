//
// Created by Jarlene on 2017/7/25.
//

#ifndef MATRIX_TIME_H
#define MATRIX_TIME_H

#include <sys/time.h>

namespace matrix {
    long getCurrentTime() {
        struct timeval tv;
        gettimeofday(&tv,NULL);
        return tv.tv_sec * 1000 + tv.tv_usec / 1000;
    }
}

#endif //MATRIX_TIME_H
