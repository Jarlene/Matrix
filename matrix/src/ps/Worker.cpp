//
// Created by Jarlene on 2018/4/23.
//

#include "matrix/include/ps/Worker.h"

namespace matrix {


    Worker::Worker() : BaseZMQ() {
        init();
    }

    Worker::~Worker() {
        BaseZMQ::~BaseZMQ();
    }


    int Worker::registerMaster() {


        return 0;
    }

    int Worker::init() {

        return 0;
    }
}