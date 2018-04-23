//
// Created by Jarlene on 2018/4/23.
//

#ifndef MATRIX_BASEZMQ_H
#define MATRIX_BASEZMQ_H

#ifndef USE_ZMQ
#pragma message("Error: Parameters servers need zeroMQ please set option use_zmp on")
#endif
#include <zmq.h>

namespace matrix {

    class BaseZMQ {
    public:
        virtual SendTo();
        virtual RecvFrom();

    };
}


#endif //MATRIX_BASEZMQ_H
