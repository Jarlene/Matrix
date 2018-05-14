//
// Created by Jarlene on 2018/4/23.
//

#include "matrix/include/ps/Worker.h"

namespace matrix {


    Worker::Worker() : BaseZMQ() {

    }

    Worker::~Worker() {
        BaseZMQ::~BaseZMQ();
    }

    int Worker::SendTo(const Addr &addr, Packet *p) {

        return 0;
    }

    int Worker::ReceiveFrom(const Addr &addr, Packet *p) {
        return 0;
    }

    int Worker::Notify() {
        return 0;
    }
}