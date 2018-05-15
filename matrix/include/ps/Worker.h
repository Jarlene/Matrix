//
// Created by Jarlene on 2018/4/23.
//

#ifndef MATRIX_WORK_H
#define MATRIX_WORK_H


#include "zmq/BaseZMQ.h"

namespace matrix {


    class Worker : public BaseZMQ{
    public:
        Worker();
        ~Worker();

        int SendTo(const Addr &addr, Packet *p) override;

        int ReceiveFrom(const Addr &addr, Packet *p) override;

        int Notify() override;

        int registerMaster();

    };

}

#endif //MATRIX_WORK_H
