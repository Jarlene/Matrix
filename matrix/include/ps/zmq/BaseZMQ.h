//
// Created by Jarlene on 2018/4/23.
//

#ifndef MATRIX_BASEZMQ_H
#define MATRIX_BASEZMQ_H


#ifndef USE_ZMQ
#pragma message("Error: Parameters servers need zeroMQ please set option use_zmp on")
#endif
#include <zmq.h>
#include <string>
#include "Packet.h"
namespace matrix {


    struct Addr {
        std::string addr;
        size_t port;

        const std::string toString() const {
            return "tcp://" + addr + std::to_string(port);
        }

    };

    class BaseZMQ {
    public:
        BaseZMQ() {
            zmq_ctx = zmq_ctx_new();
            assert(zmq_ctx != nullptr);
            zmq_skt = zmq_socket(zmq_ctx, ZMQ_PULL);
            assert(zmq_skt != nullptr);
        }

        virtual ~BaseZMQ() {
            zmq_ctx_destroy(zmq_ctx);
            if (zmq_skt) {
                zmq_close(zmq_skt);
                zmq_skt = nullptr;
            }
        }


        virtual int SendTo(const Addr &addr, Packet *p) = 0;

        virtual int ReceiveFrom(const Addr &addr, Packet *p) = 0;

        virtual int Notify() = 0;

    protected:
        int registerRouter(size_t node_id, Addr &&addr) {
            return 0;
        }


    protected:
        void * zmq_ctx;
        void * zmq_skt;
        Addr addr;

    };
}


#endif //MATRIX_BASEZMQ_H
