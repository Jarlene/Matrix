//
// Created by Jarlene on 2018/4/20.
//

#ifndef MATRIX_SERVER_H
#define MATRIX_SERVER_H

#include "zmq/BaseZMQ.h"



namespace matrix {


    class Server : public BaseZMQ {
    public:
        Server();
        ~Server();

        int SendTo() override;

        int ReceiveFrom() override;

    };
}

#endif //MATRIX_SERVER_H
