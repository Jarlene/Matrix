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
        kMSE,
        kLikelihood,
        kSoftmaxCrossEntropy
    };

    enum OutputMode {
        kSoftmax = 0,

    };

    enum ImageOrder {
        NCHW = 0,
        NHWC
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
        kAvg,
        kKMax
    };

    enum Phase {
        TRAIN = 0,
        TEST,
        PREDICTION
    };

    enum EmbeddingType {
        ONE_HOT,
        WORD_EMBEDDING
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
        MatrixType type = kInvalid;
        Phase phase;
        RunMode mode;

        static Context Default(MatrixType type = kFloat) {
            Context context;
            context.mode = kCpu;
            context.phase = TRAIN;
            context.type = type;
            return context;
        }

        static Context Test(MatrixType type = kFloat) {
            Context context;
            context.mode = kCpu;
            context.phase = TEST;
            context.type = type;
            return context;
        }

        static Context GPU() {
            Context context;
            context.mode = kGpu;
            context.phase = TRAIN;
            context.type = kFloat;
            return context;
        }

        static Context CPU() {
            Context context;
            context.mode = kCpu;
            context.phase = TRAIN;
            context.type = kFloat;
            return context;
        }
    };

}

#endif //MATRIX_MATRIXTYPE_H
