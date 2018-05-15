//
// Created by Jarlene on 2018/4/23.
//

#include "matrix/include/ps/Master.h"

namespace matrix {


    Master::Master() : BaseZMQ(){

    }

    Master::Master(const std::string config) : BaseZMQ(){

    }

    Master::~Master()  {
        BaseZMQ::~BaseZMQ();
    }

    int Master::SendTo(const Addr &addr, Packet *p) {
        int retry_count = 5;
        int ret = -1;
        while(retry_count--) {
            ret = zmq_bind(zmq_skt, addr.toString().c_str());
            if (ret == 0) {
                break;
            }
        }
        if(ret != 0) {
            return -1;
        }
        ret = -1;
        retry_count = 5;
        while(retry_count--) {
            ret = zmq_send(zmq_skt, p->header.data(), p->header.size(), ZMQ_SNDMORE);
            assert(ret == p->header.size());
            ret = zmq_send(zmq_skt, p->content.data(), p->content.size(), 0);
            if (ret == 0) {
                break;
            }
        }
        if(ret != 0) {
            return -1;
        }
        return 0;
    }

    int Master::ReceiveFrom(const Addr &addr, Packet *p) {
        int retry_count = 5;
        int ret = -1;
        while(retry_count--) {
            ret = zmq_bind(zmq_skt, addr.toString().c_str());
            if (ret == 0) {
                break;
            }
        }
        if(ret != 0) {
            return -1;
        }
        ret = -1;
        retry_count = 5;
        while(retry_count--) {
            ret = zmq_recv(zmq_skt, p->header.data(), p->header.size(), ZMQ_RCVMORE);
            assert(ret == p->header.size());
            ret = zmq_recv(zmq_skt, p->content.data(), p->content.size(), 0);
            if (ret == 0) {
                break;
            }
        }
        if(ret != 0) {
            return -1;
        }
        return 0;
    }
}