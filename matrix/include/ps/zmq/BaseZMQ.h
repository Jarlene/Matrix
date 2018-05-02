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
        BaseZMQ() {
            zmq_ctx = zmq_ctx_new();
        }

        virtual ~BaseZMQ() {
            zmq_ctx_destroy(zmq_ctx);
            if (zmq_socket) {
                zmq_close(zmq_socket);
                zmq_socket = nullptr;
            }
        }


        virtual int SendTo() = 0;

        virtual int  RecvFrom() = 0;

    protected:
        void * zmq_ctx;
        void * zmq_socket;


    };
}


#endif //MATRIX_BASEZMQ_H
