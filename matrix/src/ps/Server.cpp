//
// Created by Jarlene on 2018/4/23.
//

#include "matrix/include/ps/Server.h"


namespace matrix {


    Server::Server() : BaseZMQ() {

    }

    Server::~Server()  {
        BaseZMQ::~BaseZMQ();
    }
}