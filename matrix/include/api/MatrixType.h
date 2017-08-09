//
// Created by Jarlene on 2017/7/22.
//

#ifndef MATRIX_MATRIXTYPE_H
#define MATRIX_MATRIXTYPE_H


#include "matrix/include/base/Tensor.h"
#include "matrix/include/base/Shape.h"

namespace matrix {

    enum MatrixType {
        kInvalid = -1,
        kInt = 0,
        kLong,
        kFloat,
        kDouble
    };

    enum ActType {
        kSigmoid = 0,
        kRelu,
        kTanh
    };

    enum RunMode {
        kCpu = 0,
        kGpu
    };

    enum LossMode {
        kCrossEntropy = 0,
        kMSE
    };

    enum ApplyGradMode {
        kSGD = 0,
        kMomentum,
        kNesterov,
        kAdagrad,
        kAdadelta,
        kRMSprop,
        kAdam,
        kAdamax,
        kNadam
    };

    enum PoolType {
        kMax = 0,
        kAvg
    };

    enum Phase {
        TRAIN = 0,
        TEST,
        PREDICTION
    };

    struct CPU {
        const static RunMode mode = RunMode::kCpu;
        const static int kDevice = 0;
    };

    struct GPU {
        const static RunMode mode = RunMode::kGpu;
        const static int kDevice = 0;
    };


    struct Context {
        MatrixType type;
        Phase phase;
        RunMode mode;
    };

}

#endif //MATRIX_MATRIXTYPE_H
